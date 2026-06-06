from __future__ import annotations

from quorum.graph.nodes.analyze_axis import (
    MAX_FISCAL_YEARS,
    _trim_to_recent_fiscal_years,
)
from quorum.tools.concept_resolver import ResolvedFact


def _fact(period: str, concept: str = "us-gaap:Revenues") -> ResolvedFact:
    return ResolvedFact(
        value=1.0,
        unit="USD",
        period=period,
        accession="0000000000-00-000000",
        resolved_concept=concept,
    )


def test_drops_quarterly_periods() -> None:
    facts = [_fact("FY2024"), _fact("Q1-2024"), _fact("Q3-2023")]
    out = _trim_to_recent_fiscal_years(facts)
    assert {f.period for f in out} == {"FY2024"}


def test_keeps_most_recent_n_years() -> None:
    facts = [_fact(f"FY{y}") for y in (2019, 2020, 2021, 2022, 2023, 2024)]
    out = _trim_to_recent_fiscal_years(facts)
    assert {f.period for f in out} == {"FY2024", "FY2023", "FY2022", "FY2021"}


def test_default_window_is_four_years() -> None:
    assert MAX_FISCAL_YEARS == 4


def test_preserves_all_concepts_within_kept_years() -> None:
    facts = [
        _fact("FY2024", "us-gaap:Revenues"),
        _fact("FY2024", "us-gaap:NetIncomeLoss"),
        _fact("FY2023", "us-gaap:Revenues"),
    ]
    out = _trim_to_recent_fiscal_years(facts)
    assert len(out) == 3
    assert {(f.period, f.resolved_concept) for f in out} == {
        ("FY2024", "us-gaap:Revenues"),
        ("FY2024", "us-gaap:NetIncomeLoss"),
        ("FY2023", "us-gaap:Revenues"),
    }


def test_empty_input_returns_empty() -> None:
    assert _trim_to_recent_fiscal_years([]) == []


def test_quarterly_only_returns_empty() -> None:
    facts = [_fact("Q1-2024"), _fact("Q2-2024")]
    assert _trim_to_recent_fiscal_years(facts) == []


def test_fewer_years_than_window_kept_intact() -> None:
    facts = [_fact("FY2024"), _fact("FY2023")]
    out = _trim_to_recent_fiscal_years(facts)
    assert {f.period for f in out} == {"FY2024", "FY2023"}


def test_custom_window() -> None:
    facts = [_fact(f"FY{y}") for y in (2021, 2022, 2023, 2024)]
    out = _trim_to_recent_fiscal_years(facts, max_years=2)
    assert {f.period for f in out} == {"FY2024", "FY2023"}
