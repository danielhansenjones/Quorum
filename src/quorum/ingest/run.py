from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from diskcache import Cache
from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient

from quorum.cache.embed_cache import cached_embed_batch, open_embed_cache
from quorum.config.companies import CIK_BY_TICKER, COMPANIES, Company
from quorum.config.settings import get_settings
from quorum.ingest.aliases import populate_aliases
from quorum.ingest.chunk import Chunker, chunk_filing
from quorum.ingest.edgar import EdgarFetcher, FilingRef
from quorum.ingest.facts import iter_facts, upsert_facts
from quorum.ingest.parse import parse_filing_html
from quorum.ingest.qdrant_writer import PointPayload, ensure_collection, upsert_points
from quorum.models.embed import BGEM3Embedder
from quorum.trace.logger import configure_logging, get_logger
from quorum.trace.writer import open_pool

DEFAULT_AXES_DIR = Path("config")
DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_EDGAR_CACHE = Path("data/edgar")


def _fiscal_period_for(filing: FilingRef) -> str:
    # Lightweight derivation from report_date; the canonical period label lives
    # in the XBRL facts table (where fp + fy are exact). For text chunks we use
    # the report date's year for 10-K, and the quarter for 10-Q where derivable.
    rd = filing.report_date  # YYYY-MM-DD
    year = rd[:4]
    if filing.form == "10-K":
        return f"FY{year}"
    return f"FQ-{rd}"


def ingest_company(
    company: Company,
    *,
    fetcher: EdgarFetcher,
    embedder: BGEM3Embedder,
    qdrant: QdrantClient,
    pool: ConnectionPool,
    embed_cache: Cache,
    chunker: Chunker | None = None,
    max_10k: int = 1,
    max_10q: int = 4,
) -> dict[str, int]:
    log = get_logger("ingest")
    counts = {"filings": 0, "chunks": 0, "facts": 0}

    # 1. XBRL facts from companyfacts JSON (one network call per CIK).
    try:
        cf = fetcher.fetch_company_facts(company.cik)
        facts = list(iter_facts(company.cik, cf))
        counts["facts"] = upsert_facts(pool, facts)
        log.info("facts_upserted", ticker=company.ticker, count=counts["facts"])
    except Exception as e:
        log.warning("companyfacts_failed", ticker=company.ticker, error=str(e))

    # 2. Filings (text path). Per decisions.md #3: latest 10-K + 4 most recent 10-Qs.
    filings = fetcher.list_filings(
        company.cik, forms=("10-K", "10-Q"), max_per_form={"10-K": max_10k, "10-Q": max_10q}
    )
    counts["filings"] = len(filings)

    chunker = chunker or Chunker()
    for filing in filings:
        html = fetcher.fetch_primary_doc(filing)
        sections = parse_filing_html(html, form=filing.form)
        chunks = chunk_filing(sections, chunker=chunker)
        if not chunks:
            continue

        texts = [c.text for c in chunks]
        # Hot path: cache embeddings per text. A re-ingest hits 100%.
        dense_vecs_arr = cached_embed_batch(
            embed_cache,
            model_name="bge-m3-dense",
            texts=texts,
            embed_fn=lambda t: embedder.embed(list(t), return_sparse=False)["dense_vecs"].tolist(),
        )
        sparse_weights = cached_embed_batch(
            embed_cache,
            model_name="bge-m3-sparse",
            texts=texts,
            embed_fn=lambda t: embedder.embed(list(t), return_dense=False)["lexical_weights"],
        )

        fiscal_period = _fiscal_period_for(filing)
        payloads = [
            PointPayload(
                chunk_id=c.chunk_id(filing.accession),
                ticker=company.ticker,
                cik=company.cik,
                accession=filing.accession,
                form=filing.form,
                section=c.section,
                fiscal_period=fiscal_period,
                filing_date=filing.filing_date,
                char_start=c.char_start,
                char_end=c.char_end,
                text=c.text,
            )
            for c in chunks
        ]
        n = upsert_points(
            qdrant, payloads=payloads, dense_vecs=dense_vecs_arr, sparse_weights=sparse_weights
        )
        counts["chunks"] += n
        log.info(
            "filing_ingested",
            ticker=company.ticker,
            accession=filing.accession,
            form=filing.form,
            chunks=n,
        )

    return counts


def ingest(tickers: Iterable[str] | None = None) -> dict[str, dict[str, int]]:
    configure_logging()
    settings = get_settings()
    log = get_logger("ingest")
    log.info("ingest_start", tickers=list(tickers) if tickers else "all")

    targets: tuple[Company, ...] = (
        tuple(c for c in COMPANIES if c.ticker in set(tickers)) if tickers else COMPANIES
    )

    qdrant = QdrantClient(url=settings.qdrant_url)
    ensure_collection(qdrant)
    pool = open_pool(
        conninfo=settings.postgres_url,
        min_size=settings.pg_pool_min,
        max_size=settings.pg_pool_max,
    )
    populate_aliases(pool, DEFAULT_AXES_DIR / "concept_aliases.yaml")

    embedder = BGEM3Embedder(device="cpu")
    embed_cache = open_embed_cache(settings.cache_dir / "embeddings")
    fetcher = EdgarFetcher(user_agent=settings.edgar_user_agent, cache_dir=DEFAULT_EDGAR_CACHE)

    results: dict[str, dict[str, int]] = {}
    try:
        for company in targets:
            results[company.ticker] = ingest_company(
                company,
                fetcher=fetcher,
                embedder=embedder,
                qdrant=qdrant,
                pool=pool,
                embed_cache=embed_cache,
            )
    finally:
        fetcher.close()
        pool.close()

    log.info("ingest_complete", results=results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Quorum offline ingestion (Phase 3)")
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers to ingest. Empty = all v1 companies.",
    )
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None
    if tickers:
        unknown = [t for t in tickers if t not in CIK_BY_TICKER]
        if unknown:
            print(f"Unknown tickers: {unknown}", flush=True)
            return 2
    ingest(tickers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
