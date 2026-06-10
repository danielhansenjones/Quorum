from __future__ import annotations

import pytest


def test_build_mcp_server_lists_all_tools() -> None:
    # Phase 8 gate: the MCP server exposes all 7 tools (6 primitives + the
    # high-level compare_companies). We don't run a live MCP inspector here;
    # we introspect the FastMCP instance directly.
    from quorum.mcp.server import build_mcp_server

    server = build_mcp_server(
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        postgres_conninfo="postgresql://noop",
        embed_query=lambda q: ([0.0], {}),
        compiled_graph=None,
    )

    # FastMCP keeps tools in a private registry; the public surface is the
    # `list_tools()` async method on the underlying server. For a sync probe
    # we touch the internal `tool_manager`.
    tm = getattr(server, "_tool_manager", None) or getattr(server, "tool_manager", None)
    assert tm is not None, "FastMCP tool manager not found via known attrs"
    tools = list(tm.list_tools())
    names = {t.name for t in tools}
    expected = {
        "resolve_company",
        "search_filings",
        "get_financial_concept_tool",
        "get_filing_section_tool",
        "list_corpus_tool",
        "list_filings_tool",
        "compare_companies",
    }
    assert expected.issubset(names), f"missing tools: {expected - names}"


def test_search_filings_rejects_empty_query() -> None:
    from quorum.mcp.server import build_mcp_server

    server = build_mcp_server(
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        postgres_conninfo="postgresql://noop",
        embed_query=lambda q: ([0.0], {}),
        compiled_graph=None,
    )
    tm = getattr(server, "_tool_manager", None) or getattr(server, "tool_manager", None)
    assert tm is not None
    tools = {t.name: t for t in tm.list_tools()}
    sf = tools["search_filings"]
    # FastMCP wraps the underlying callable as `fn`; surface the validator.
    fn = getattr(sf, "fn", None)
    assert fn is not None
    with pytest.raises(ValueError):
        fn(query="   ")
