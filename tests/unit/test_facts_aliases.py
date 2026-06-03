from __future__ import annotations

from datetime import date
from pathlib import Path

from quorum.ingest.aliases import (
    DEFAULT_TICKER_TOKEN,
    expand_aliases,
    load_aliases_yaml,
)
from quorum.ingest.facts import (
    _classify_period,
    _fiscal_quarter,
    _fiscal_year,
    _infer_fy_end_month,
    iter_facts,
)


def test_fiscal_year_calendar_company() -> None:
    # KO: fy_end_month=12. End-of-year always maps to its own calendar year.
    assert _fiscal_year(date(2024, 12, 31), fy_end_month=12) == 2024
    assert _fiscal_year(date(2024, 3, 31), fy_end_month=12) == 2024


def test_fiscal_year_apple_september_end() -> None:
    # AAPL: fy_end_month=9. A quarter ending after September belongs to next FY.
    assert _fiscal_year(date(2018, 12, 29), fy_end_month=9) == 2019  # Q1 FY2019
    assert _fiscal_year(date(2019, 3, 30), fy_end_month=9) == 2019  # Q2 FY2019
    assert _fiscal_year(date(2019, 6, 29), fy_end_month=9) == 2019  # Q3 FY2019
    assert _fiscal_year(date(2019, 9, 28), fy_end_month=9) == 2019  # Q4 / FY2019


def test_fiscal_quarter_apple() -> None:
    # AAPL fiscal year ends September. Q4 ends at FY end; Q1-Q3 follow.
    assert _fiscal_quarter(date(2019, 9, 28), fy_end_month=9) == "Q4"
    assert _fiscal_quarter(date(2018, 12, 29), fy_end_month=9) == "Q1"
    assert _fiscal_quarter(date(2019, 3, 30), fy_end_month=9) == "Q2"
    assert _fiscal_quarter(date(2019, 6, 29), fy_end_month=9) == "Q3"


def test_fiscal_quarter_calendar_year() -> None:
    # KO / PEP / Pharma cohort: fy_end_month=12.
    assert _fiscal_quarter(date(2024, 3, 31), fy_end_month=12) == "Q1"
    assert _fiscal_quarter(date(2024, 12, 31), fy_end_month=12) == "Q4"


def test_classify_annual_uses_end_year_not_fy() -> None:
    # The bug fix: companyfacts often labels prior-period comparatives with the
    # restating filing's fy. End date is the truth.
    dp = {"start": "2017-10-01", "end": "2018-09-29", "fp": "FY", "fy": 2019, "val": 1, "accn": "x"}
    assert _classify_period(dp, fy_end_month=9) == "FY2018"


def test_classify_annual_apple_fy2019() -> None:
    # The canonical AAPL FY2019 datapoint must land at "FY2019".
    dp = {"start": "2018-09-30", "end": "2019-09-28", "fp": "FY", "fy": 2019, "val": 1, "accn": "x"}
    assert _classify_period(dp, fy_end_month=9) == "FY2019"


def test_classify_drops_quarterly_mislabeled_fy() -> None:
    # AAPL Q4 2019 standalone slice ($64B). companyfacts tags it fp=FY because
    # it comes from a 10-K. Our classifier must reroute by duration to "Q4-2019",
    # NOT collapse it into "FY2019".
    dp = {"start": "2019-06-30", "end": "2019-09-28", "fp": "FY", "fy": 2019, "val": 1, "accn": "x"}
    assert _classify_period(dp, fy_end_month=9) == "Q4-2019"


def test_classify_instant_uses_fp_fy() -> None:
    # Balance sheet items have no start. The SEC fp/fy is reliable for these.
    dp = {"end": "2024-12-31", "fp": "FY", "fy": 2024, "val": 1, "accn": "x"}
    assert _classify_period(dp, fy_end_month=None) == "FY2024"


def test_classify_drops_off_cycle_durations() -> None:
    # 6-month YTD slice from a 10-Q comparative. Neither annual nor quarterly.
    dp = {"start": "2024-01-01", "end": "2024-06-30", "fp": "Q2", "fy": 2024, "val": 1, "accn": "x"}
    assert _classify_period(dp, fy_end_month=12) is None


def test_infer_fy_end_month_apple() -> None:
    cf = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2018-09-30",
                                "end": "2019-09-28",
                                "fp": "FY",
                                "fy": 2019,
                                "val": 1,
                                "accn": "a",
                            },
                            {
                                "start": "2019-09-29",
                                "end": "2020-09-26",
                                "fp": "FY",
                                "fy": 2020,
                                "val": 1,
                                "accn": "b",
                            },
                            {
                                "start": "2018-09-30",
                                "end": "2018-12-29",
                                "fp": "Q1",
                                "fy": 2019,
                                "val": 1,
                                "accn": "c",
                            },
                        ]
                    }
                }
            }
        }
    }
    assert _infer_fy_end_month(cf) == 9


def test_iter_facts_aapl_fy2019_regression() -> None:
    # Reproduces the production canary bug: companyfacts has both the
    # consolidated FY2019 ($260B) and the standalone Q4 2019 slice ($64B)
    # under fp=FY fy=2019. Old code returned $64B for the FY2019 query;
    # the fix must yield exactly one FY2019 fact with value $260B.
    cf = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2018-09-30",
                                "end": "2019-09-28",
                                "fp": "FY",
                                "fy": 2019,
                                "val": 260_174_000_000,
                                "accn": "acc-2019",
                            },
                            {
                                "start": "2018-09-30",
                                "end": "2019-09-28",
                                "fp": "FY",
                                "fy": 2020,
                                "val": 260_174_000_000,
                                "accn": "acc-2020",
                            },
                            {
                                "start": "2019-06-30",
                                "end": "2019-09-28",
                                "fp": "FY",
                                "fy": 2019,
                                "val": 64_040_000_000,
                                "accn": "acc-2019",
                            },
                            {
                                "start": "2017-10-01",
                                "end": "2018-09-29",
                                "fp": "FY",
                                "fy": 2019,
                                "val": 265_595_000_000,
                                "accn": "acc-2019",
                            },
                        ]
                    }
                }
            }
        }
    }
    rows = list(iter_facts("0000320193", cf))
    by_period = {r.period: r for r in rows}
    assert "FY2019" in by_period
    assert by_period["FY2019"].value == 260_174_000_000
    assert "FY2018" in by_period
    assert by_period["FY2018"].value == 265_595_000_000
    assert "Q4-2019" in by_period
    assert by_period["Q4-2019"].value == 64_040_000_000


def test_iter_facts_dedup_keeps_latest_accession() -> None:
    # Same fact, two accessions (original + later restating filing).
    # Latest accn wins because it carries the most recent restated value.
    cf = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "fp": "FY",
                                "fy": 2023,
                                "val": 100,
                                "accn": "0000000000-23-000001",
                            },
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "fp": "FY",
                                "fy": 2024,
                                "val": 100,
                                "accn": "0000000000-24-000001",
                            },
                        ]
                    }
                }
            }
        }
    }
    rows = list(iter_facts("0", cf))
    assert len(rows) == 1
    assert rows[0].accession == "0000000000-24-000001"


def test_iter_facts_skips_incomplete() -> None:
    cf = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 1000,
                                "accn": "acc-1",
                                "fp": "FY",
                                "fy": 2025,
                                "end": "2025-09-30",
                            },
                            {
                                "val": 800,
                                "accn": "acc-2",
                                "fp": "Q3",
                                "fy": 2025,
                                "end": "2025-06-30",
                            },
                            # Missing fp/fy: should be skipped.
                            {"val": 99, "accn": "acc-3"},
                            # Bogus fp: should be skipped (no period label).
                            {"val": 7, "accn": "acc-4", "fp": "BOGUS", "fy": 2025},
                        ]
                    }
                }
            }
        }
    }
    rows = list(iter_facts("320193", cf))
    periods = sorted(r.period for r in rows)
    assert periods == ["FY2025", "Q3-2025"]


def test_iter_facts_includes_unit() -> None:
    # NetIncomeLoss as an instant snapshot (no start). Real SEC data always
    # carries at least an end date for these.
    cf = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 93.7e9,
                                "accn": "acc",
                                "fp": "FY",
                                "fy": 2025,
                                "end": "2025-09-27",
                            }
                        ]
                    }
                }
            }
        }
    }
    rows = list(iter_facts("320193", cf))
    assert rows[0].unit == "USD"
    assert rows[0].concept == "us-gaap:NetIncomeLoss"


def test_expand_aliases_default_chain() -> None:
    aliases = {
        "profitability.revenue": {
            "default": ["us-gaap:Revenues", "us-gaap:SalesRevenueNet"],
            "per_ticker": {
                "AAPL": ["us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"],
            },
        }
    }
    rows = expand_aliases(aliases)
    default_rows = [r for r in rows if r[1] == DEFAULT_TICKER_TOKEN]
    aapl_rows = [r for r in rows if r[1] == "AAPL"]
    assert len(default_rows) == 2
    assert default_rows[0][2] == 0  # ordering preserved
    assert default_rows[1][2] == 1
    assert len(aapl_rows) == 1
    assert aapl_rows[0][3] == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"


def test_load_real_aliases_yaml() -> None:
    # The committed config/concept_aliases.yaml must be valid and cover at least
    # the profitability axis for the v1 corpus.
    root = Path(__file__).resolve().parents[2]
    path = root / "config" / "concept_aliases.yaml"
    aliases = load_aliases_yaml(path)
    assert "profitability.revenue" in aliases
    assert "us-gaap:Revenues" in aliases["profitability.revenue"]["default"]
    # AAPL must have a per-ticker override (the canonical "Apple uses 606" example).
    assert "AAPL" in aliases["profitability.revenue"]["per_ticker"]
