from __future__ import annotations

import re
import socket

import pytest

from quorum.config.companies import CIK_BY_TICKER
from quorum.graph.axis_config import AXIS_CONCEPTS
from quorum.tools.concept_resolver import get_financial_concept
from quorum.trace.writer import open_pool

pytestmark = pytest.mark.integration


def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture
def pool(postgres_url: str):
    if not _tcp_reachable("localhost", 5432):
        pytest.skip("postgres not reachable")
    p = open_pool(conninfo=postgres_url, min_size=1, max_size=4)
    try:
        yield p
    finally:
        p.close()


def test_aapl_fy2024_revenue_matches_ground_truth(pool) -> None:
    # Phase 3e: AAPL FY2024 revenue is publicly reported at $391.04B (Form 10-K
    # filed 2024-11-01, accession 0000320193-24-000123). Tolerance is 1% in case
    # of restated/rounded values from companyfacts.
    rows = get_financial_concept(
        pool, ticker="AAPL", key="profitability.revenue", periods=["FY2024"]
    )
    assert len(rows) == 1
    fact = rows[0]
    assert fact.period == "FY2024"
    assert fact.unit == "USD"
    assert fact.resolved_concept == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert fact.value == pytest.approx(391.035e9, rel=0.01)


def test_cross_company_resolver_canary(pool) -> None:
    # Phase 3e: AAPL and KO use different XBRL concepts for revenue.
    # AAPL's per-ticker override picks RevenueFromContractWithCustomer*;
    # KO has no override and falls to the default Revenues chain.
    aapl = get_financial_concept(
        pool, ticker="AAPL", key="profitability.revenue", periods=["FY2024"]
    )
    ko = get_financial_concept(pool, ticker="KO", key="profitability.revenue", periods=["FY2024"])
    assert aapl and ko, f"empty results: aapl={aapl} ko={ko}"
    assert aapl[0].resolved_concept != ko[0].resolved_concept
    assert aapl[0].resolved_concept == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert ko[0].resolved_concept == "us-gaap:Revenues"


# (ticker, key) pairs with no recent-FY value because the company genuinely does
# not tag that concept - not an alias bug. Verified against companyfacts 2026-07-05:
# tech/pharma rarely tag GrossProfit; pharma structures its income statement
# without OperatingIncomeLoss; minimal-debt names (META) and bundled-current-debt
# filers (PEP/LLY report only DebtCurrent, which overlaps commercial paper and so
# is excluded to avoid double counting) lack the current/short-term debt pieces.
_KNOWN_EMPTY: frozenset[tuple[str, str]] = frozenset(
    {
        ("GOOGL", "profitability.gross_profit"),
        ("META", "profitability.gross_profit"),
        ("META", "leverage.current_debt"),
        ("META", "leverage.short_term_debt"),
        ("PG", "profitability.gross_profit"),
        ("PEP", "leverage.current_debt"),
        ("COST", "profitability.gross_profit"),
        ("COST", "leverage.short_term_debt"),
        ("JNJ", "profitability.operating_income"),
        ("PFE", "profitability.operating_income"),
        ("PFE", "profitability.gross_profit"),
        ("MRK", "profitability.operating_income"),
        ("MRK", "profitability.gross_profit"),
        ("MRK", "leverage.short_term_debt"),
        ("LLY", "profitability.operating_income"),
        ("LLY", "profitability.gross_profit"),
        ("LLY", "leverage.current_debt"),
    }
)

_FY_RE = re.compile(r"^FY(\d{4})$")


def test_systematic_resolver_canary(pool) -> None:
    # Phase 12g: every (ticker x axis-concept the analyst pulls) resolves to a
    # recent fiscal year except a documented set of genuine not-reported concepts.
    # This turns an alias gap - which otherwise surfaces only as a low faithfulness
    # score on one eval case - into a test failure. It also fails if a known gap
    # silently closes, keeping the exception list honest.
    keys = sorted({k for concepts in AXIS_CONCEPTS.values() for k in concepts})
    actual_empty: set[tuple[str, str]] = set()
    for ticker in CIK_BY_TICKER:
        for key in keys:
            rows = get_financial_concept(pool, ticker=ticker, key=key)
            recent = [r for r in rows if (m := _FY_RE.match(r.period)) and int(m.group(1)) >= 2021]
            if not recent:
                actual_empty.add((ticker, key))
    new_gaps = sorted(actual_empty - _KNOWN_EMPTY)
    closed = sorted(_KNOWN_EMPTY - actual_empty)
    assert not new_gaps and not closed, (
        "resolver coverage drifted from the documented baseline.\n"
        f"  new alias gaps (regression, fix the chain): {new_gaps}\n"
        f"  gaps that now resolve (drop from _KNOWN_EMPTY): {closed}"
    )


def test_period_filter_narrows_results(pool) -> None:
    # Phase 4b: period filter is strict. Asking for FY2024 must not return
    # rows for any other period.
    rows = get_financial_concept(
        pool, ticker="AAPL", key="profitability.revenue", periods=["FY2024"]
    )
    assert rows, "no rows returned"
    for r in rows:
        assert r.period == "FY2024"


def test_period_filter_multi_period(pool) -> None:
    # Phase 4b: multi-period filter returns one row per requested period.
    periods = ["FY2024", "FY2023", "FY2022"]
    rows = get_financial_concept(pool, ticker="AAPL", key="profitability.revenue", periods=periods)
    returned = sorted(r.period for r in rows)
    assert returned == sorted(periods), f"got {returned}"


def test_every_returned_row_has_non_null_unit(pool) -> None:
    # Phase 4b: NOT NULL constraint at DDL plus the resolver returning unit
    # verbatim. Probe across all 12 tickers + 4 axes for broad coverage.
    tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "PG",
        "KO",
        "PEP",
        "COST",
        "JNJ",
        "PFE",
        "MRK",
        "LLY",
    ]
    keys = [
        "profitability.revenue",
        "profitability.gross_profit",
        "leverage.long_term_debt",
        "growth.revenue",
    ]
    seen = 0
    for ticker in tickers:
        for key in keys:
            rows = get_financial_concept(pool, ticker=ticker, key=key)
            for r in rows:
                assert r.unit, f"empty unit for {ticker}/{key} period={r.period}"
                seen += 1
    assert seen > 0, "no rows resolved across the entire matrix"


def test_normalized_key_works_for_core_tickers_across_axes(pool) -> None:
    # Phase 4b: at least 3 axes return non-empty results for the BigTech +
    # Staples subset. Pharma is excluded only because some growth metrics on
    # Pharma require alternate aliases we have not landed yet.
    tickers = ["AAPL", "MSFT", "GOOGL", "KO"]
    keys = ["profitability.revenue", "profitability.net_income", "leverage.long_term_debt"]
    misses: list[tuple[str, str]] = []
    for ticker in tickers:
        for key in keys:
            rows = get_financial_concept(pool, ticker=ticker, key=key)
            if not rows:
                misses.append((ticker, key))
    assert not misses, f"normalized key resolved empty for: {misses}"


def test_unknown_concept_returns_empty(pool) -> None:
    # Phase 4b: unknown axis_metric_key returns [] without raising.
    rows = get_financial_concept(
        pool, ticker="AAPL", key="bogus.does_not_exist", periods=["FY2024"]
    )
    assert rows == []


def test_unknown_ticker_returns_empty(pool) -> None:
    rows = get_financial_concept(
        pool, ticker="NOPE", key="profitability.revenue", periods=["FY2024"]
    )
    assert rows == []
