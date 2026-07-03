from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP
from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient

from quorum.models.router import ChatClient
from quorum.tools.concept_resolver import get_financial_concept
from quorum.tools.filing_section import FilingSectionNotFound, get_filing_section
from quorum.tools.inventory import list_corpus as _list_corpus
from quorum.tools.inventory import list_filings as _list_filings
from quorum.tools.resolve_company import resolve_company as _resolve_company
from quorum.tools.search import hybrid_search


def build_mcp_server(
    *,
    pool: ConnectionPool,
    qdrant: QdrantClient,
    postgres_conninfo: str,
    embed_query: Callable[[str], tuple[list[float], dict[str, float]]],
    classifier_client: ChatClient | None = None,
    sonnet_client: ChatClient | None = None,
    compiled_graph: Any | None = None,
) -> FastMCP:
    # Phase 8: every Phase 4 tool is also an MCP tool. `compare_companies` is
    # the high-level capability that wraps the compiled LangGraph.
    server = FastMCP("quorum")

    @server.tool()
    def resolve_company(query: str) -> dict[str, Any]:
        """Resolve a company name/ticker/CIK to a canonical record."""
        rc = _resolve_company(query)
        if rc is None:
            return {"status": "not_found"}
        return {"status": "ok", "ticker": rc.ticker, "cik": rc.cik, "name": rc.name}

    @server.tool()
    def search_filings(
        query: str,
        tickers: list[str] | None = None,
        sections: list[str] | None = None,
        forms: list[str] | None = None,
        fiscal_periods: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid retrieval over filing chunks. Returns dense+sparse-fused hits."""
        if not query.strip():
            raise ValueError("query must be non-empty")
        dense, sparse = embed_query(query)
        hits = hybrid_search(
            qdrant,
            dense_vec=dense,
            sparse_weights=sparse,
            tickers=tickers,
            sections=sections,
            forms=forms,
            fiscal_periods=fiscal_periods,
            top_k=top_k,
        )
        return [
            {
                "chunk_id": h.chunk_id,
                "score": h.score,
                "ticker": h.payload.get("ticker"),
                "accession": h.payload.get("accession"),
                "section": h.payload.get("section"),
                "fiscal_period": h.payload.get("fiscal_period"),
                "text": h.payload.get("text"),
            }
            for h in hits
        ]

    @server.tool()
    def get_financial_concept_tool(
        ticker: str,
        key: str,
        periods: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve a normalized concept key (e.g. profitability.revenue) or raw XBRL concept."""
        facts = get_financial_concept(pool, ticker=ticker, key=key, periods=periods)
        return [
            {
                "value": f.value,
                "unit": f.unit,
                "period": f.period,
                "accession": f.accession,
                "resolved_concept": f.resolved_concept,
            }
            for f in facts
        ]

    @server.tool()
    def get_filing_section_tool(ticker: str, accession: str, section: str) -> dict[str, Any]:
        """Return the full text of one section of one filing."""
        try:
            s = get_filing_section(qdrant, ticker=ticker, accession=accession, section=section)
        except FilingSectionNotFound as e:
            raise ValueError(str(e)) from e
        return {
            "ticker": s.ticker,
            "accession": s.accession,
            "section": s.section,
            "text": s.text,
            "chunk_ids": s.chunk_ids,
        }

    @server.tool()
    def list_corpus_tool() -> dict[str, Any]:
        """Cross-checked inventory of facts (Postgres) and chunks (Qdrant) per company."""
        entries = _list_corpus(qdrant, postgres_conninfo)
        return {
            "companies": [
                {
                    "ticker": e.ticker,
                    "cik": e.cik,
                    "facts_count": e.facts_count,
                    "chunks_count": e.chunks_count,
                }
                for e in entries
            ]
        }

    @server.tool()
    def list_filings_tool(ticker: str | None = None) -> list[dict[str, Any]]:
        """List filings present in the corpus, optionally filtered by ticker."""
        summaries = _list_filings(qdrant, ticker=ticker)
        return [
            {
                "ticker": s.ticker,
                "accession": s.accession,
                "form": s.form,
                "fiscal_period": s.fiscal_period,
                "filing_date": s.filing_date,
                "chunk_count": s.chunk_count,
            }
            for s in summaries
        ]

    @server.tool()
    def compare_companies(question: str, max_replans: int = 2) -> dict[str, Any]:
        """Run the full Quorum graph: classify -> resolve -> plan -> analyze -> critic -> synthesize."""
        if compiled_graph is None:
            raise RuntimeError(
                "compare_companies requires a compiled graph; pass one to build_mcp_server"
            )
        from quorum.graph.build import initial_state

        state = initial_state(question, max_replans=max_replans)
        final = compiled_graph.invoke(state)
        return {
            "report": getattr(final, "report", None) or final.get("report"),
            "citations": [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in (
                    getattr(final, "report_citations", None) or final.get("report_citations", [])
                )
            ],
            "status": getattr(final, "status", None) or final.get("status"),
            "axes_analyzed": getattr(final, "axes", None) or final.get("axes", []),
            "tickers": getattr(final, "tickers", None) or final.get("tickers", []),
            "trace_id": getattr(final, "trace_id", None) or final.get("trace_id"),
        }

    return server
