from __future__ import annotations

from quorum.eval.tool_use import validate_tool_call, validate_tool_calls
from quorum.state.critique import ToolCallRecord


def _rec(tool: str, args: dict) -> ToolCallRecord:
    return ToolCallRecord(tool=tool, args=args, ok=True, result_summary="")


def test_valid_get_financial_concept() -> None:
    assert (
        validate_tool_call(
            _rec("get_financial_concept", {"ticker": "AAPL", "key": "profitability.revenue"})
        )
        == []
    )


def test_get_financial_concept_bad_ticker() -> None:
    issues = validate_tool_call(_rec("get_financial_concept", {"ticker": "TSLA", "key": "x"}))
    assert any("not in corpus" in i for i in issues)


def test_get_financial_concept_empty_key() -> None:
    issues = validate_tool_call(_rec("get_financial_concept", {"ticker": "AAPL", "key": ""}))
    assert "empty concept key" in issues


def test_valid_search_filings() -> None:
    assert (
        validate_tool_call(
            _rec("search_filings", {"query": "supply chain risk", "tickers": ["JNJ"]})
        )
        == []
    )


def test_search_filings_empty_query() -> None:
    assert "empty query" in validate_tool_call(_rec("search_filings", {"query": "  "}))


def test_search_filings_out_of_corpus_ticker() -> None:
    issues = validate_tool_call(_rec("search_filings", {"query": "q", "tickers": ["NVDA"]}))
    assert any("not in corpus" in i for i in issues)


def test_valid_get_filing_section() -> None:
    rec = _rec(
        "get_filing_section",
        {"ticker": "MSFT", "accession": "0000789019-24-000001", "section": "item_1a_risk_factors"},
    )
    assert validate_tool_call(rec) == []


def test_get_filing_section_missing_fields() -> None:
    issues = validate_tool_call(_rec("get_filing_section", {"ticker": "MSFT"}))
    assert "missing accession" in issues
    assert "missing section" in issues


def test_unknown_tool() -> None:
    assert any("unknown tool" in i for i in validate_tool_call(_rec("delete_everything", {})))


def test_aggregate() -> None:
    records = [
        _rec("get_financial_concept", {"ticker": "AAPL", "key": "profitability.revenue"}),
        _rec("get_financial_concept", {"ticker": "TSLA", "key": "x"}),
        _rec("search_filings", {"query": "risk"}),
    ]
    out = validate_tool_calls(records)
    assert out["n"] == 3
    assert out["valid"] == 2
    assert abs(out["valid_fraction"] - 2 / 3) < 1e-9


def test_aggregate_empty_is_fully_valid() -> None:
    out = validate_tool_calls([])
    assert out["n"] == 0
    assert out["valid_fraction"] == 1.0
