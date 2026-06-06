from __future__ import annotations

from quorum.graph.nodes.analyze_axis import _to_result
from quorum.state.axis import AxisTask
from quorum.tools.concept_resolver import ResolvedFact


def _task() -> AxisTask:
    return AxisTask(
        axis="profitability",
        mode="structured",
        tickers=["AAPL", "MSFT"],
        query_or_concept="profitability.revenue",
    )


def _fact(value: float, period: str) -> ResolvedFact:
    return ResolvedFact(
        value=value,
        unit="USD",
        period=period,
        accession="0000320193-24-000123",
        resolved_concept="us-gaap:Revenues",
    )


def test_assessment_comes_from_model_per_company() -> None:
    parsed = {
        "comparison": "AAPL outgrew MSFT [AAPL:Q0].",
        "per_company": {
            "AAPL": "Apple revenue rose to ~$391B [AAPL:Q0].",
            "MSFT": "Microsoft grew steadily [MSFT:Q0].",
        },
        "grounding": "ok",
    }
    quant = {"AAPL": [_fact(391.0, "FY2024")], "MSFT": [_fact(245.0, "FY2024")]}
    res = _to_result(_task(), parsed, quant, {}, attempts=1)
    assert res.per_company["AAPL"].assessment == "Apple revenue rose to ~$391B [AAPL:Q0]."
    assert res.per_company["MSFT"].assessment == "Microsoft grew steadily [MSFT:Q0]."


def test_values_are_code_built_not_model_echoed() -> None:
    parsed = {
        "comparison": "c",
        # Model tries to smuggle a wrong number; it must be ignored for values.
        "per_company": {"AAPL": "revenue was $999B [AAPL:Q0]"},
        "grounding": "ok",
    }
    quant = {"AAPL": [_fact(391.0, "FY2024")], "MSFT": []}
    res = _to_result(_task(), parsed, quant, {}, attempts=1)
    assert res.per_company["AAPL"].values == {"us-gaap:Revenues_FY2024": "391.0 USD"}
    assert "999" not in str(res.per_company["AAPL"].values)


def test_missing_per_company_entry_yields_empty_assessment() -> None:
    parsed = {"comparison": "c", "per_company": {"AAPL": "only apple"}, "grounding": "weak"}
    quant = {"AAPL": [_fact(391.0, "FY2024")], "MSFT": [_fact(245.0, "FY2024")]}
    res = _to_result(_task(), parsed, quant, {}, attempts=1)
    assert res.per_company["MSFT"].assessment == ""


def test_per_company_absent_entirely_is_tolerated() -> None:
    parsed = {"comparison": "c", "grounding": "ok"}
    quant = {"AAPL": [_fact(391.0, "FY2024")], "MSFT": [_fact(245.0, "FY2024")]}
    res = _to_result(_task(), parsed, quant, {}, attempts=1)
    assert res.per_company["AAPL"].assessment == ""
    assert res.per_company["AAPL"].values == {"us-gaap:Revenues_FY2024": "391.0 USD"}
