from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any

from psycopg_pool import ConnectionPool

# Duration windows. SEC fiscal years are 52 or 53 weeks (364 or 371 days);
# fiscal quarters are 13 weeks (91 days). Widen slightly for off-by-day variance.
_ANNUAL_MIN_DAYS = 350
_ANNUAL_MAX_DAYS = 380
_QUARTERLY_MIN_DAYS = 80
_QUARTERLY_MAX_DAYS = 100

# Map "months since fiscal year end" to quarter token. Q4 ends at the FY end
# itself (delta 0); Q1 ends 3 months later; Q2 at 6; Q3 at 9.
_QUARTER_BY_DELTA: dict[int, str] = {0: "Q4", 3: "Q1", 6: "Q2", 9: "Q3"}


@dataclass(frozen=True, slots=True)
class Fact:
    cik: str
    concept: str
    period: str
    value: float
    unit: str
    accession: str


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _infer_fy_end_month(companyfacts: dict[str, Any]) -> int | None:
    # Scan annual-duration datapoints across all concepts; the dominant
    # end-month is the company's fiscal year-end. Returns None if no annual
    # facts are present (rare; fall back to trusting fp/fy in that case).
    months: Counter[int] = Counter()
    for _taxonomy, concepts in companyfacts.get("facts", {}).items():
        for _concept, body in concepts.items():
            for _unit, datapoints in body.get("units", {}).items():
                for dp in datapoints:
                    s = _parse_date(dp.get("start"))
                    e = _parse_date(dp.get("end"))
                    if s is None or e is None:
                        continue
                    dur = (e - s).days
                    if _ANNUAL_MIN_DAYS <= dur <= _ANNUAL_MAX_DAYS:
                        months[e.month] += 1
    if not months:
        return None
    return months.most_common(1)[0][0]


def _fiscal_year(end: date, fy_end_month: int) -> int:
    # A quarter whose end-month falls AFTER the fiscal year-end month belongs
    # to the next fiscal year. For all v1 corpus companies the FY label
    # matches end.year when end.month <= fy_end_month.
    if end.month > fy_end_month:
        return end.year + 1
    return end.year


def _fiscal_quarter(end: date, fy_end_month: int) -> str | None:
    delta = (end.month - fy_end_month) % 12
    return _QUARTER_BY_DELTA.get(delta)


def _classify_period(dp: dict[str, Any], fy_end_month: int | None) -> str | None:
    end = _parse_date(dp.get("end"))
    if end is None:
        return None

    start = _parse_date(dp.get("start"))

    if start is None:
        # Instant fact (balance sheet snapshot). Trust fp/fy; the SEC's
        # labeling is reliable for instant facts because there's no
        # duration ambiguity.
        fp = dp.get("fp")
        fy = dp.get("fy")
        if fy is None:
            return None
        if fp == "FY":
            return f"FY{fy}"
        if fp in ("Q1", "Q2", "Q3", "Q4"):
            return f"{fp}-{fy}"
        return None

    duration = (end - start).days

    if _ANNUAL_MIN_DAYS <= duration <= _ANNUAL_MAX_DAYS:
        return f"FY{end.year}"

    if _QUARTERLY_MIN_DAYS <= duration <= _QUARTERLY_MAX_DAYS:
        if fy_end_month is None:
            fp = dp.get("fp")
            fy = dp.get("fy")
            if fp in ("Q1", "Q2", "Q3", "Q4") and fy is not None:
                return f"{fp}-{fy}"
            return None
        quarter = _fiscal_quarter(end, fy_end_month)
        if quarter is None:
            return None
        return f"{quarter}-{_fiscal_year(end, fy_end_month)}"

    # Half-year, year-to-date cumulative, or other off-cycle slice; drop.
    return None


def iter_facts(cik: str, companyfacts: dict[str, Any]) -> Iterator[Fact]:
    # companyfacts schema: facts.{taxonomy}.{concept}.units.{unit}[]
    # The SEC feed re-publishes each fact under every filing that includes it
    # as a comparative (one row per restating filing, same value, different
    # accession + different fp/fy metadata). It also stamps Q4 quarterly
    # slices with fp=FY when they come from a 10-K. The (start, end, duration)
    # tuple is the only reliable discriminator; dp.fp/dp.fy describe the
    # filing, not the fact's real fiscal period.
    facts_block = companyfacts.get("facts", {})
    fy_end_month = _infer_fy_end_month(companyfacts)

    # Dedup key: (taxonomy:concept, period, unit). Multiple datapoints can map
    # to the same key (the restatement clones). Keep the one with the latest
    # accession - most recent restatement = most authoritative value.
    chosen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for taxonomy, concepts in facts_block.items():
        for concept, body in concepts.items():
            fq_concept = f"{taxonomy}:{concept}"
            for unit, datapoints in body.get("units", {}).items():
                for dp in datapoints:
                    if "val" not in dp or "accn" not in dp:
                        continue
                    period = _classify_period(dp, fy_end_month)
                    if period is None:
                        continue
                    key = (fq_concept, period, unit)
                    accn = str(dp["accn"])
                    existing = chosen.get(key)
                    if existing is None or accn > str(existing["accn"]):
                        chosen[key] = {"val": dp["val"], "accn": accn}

    for (fq_concept, period, unit), row in chosen.items():
        yield Fact(
            cik=cik,
            concept=fq_concept,
            period=period,
            value=float(row["val"]),
            unit=unit,
            accession=str(row["accn"]),
        )


_UPSERT = """
INSERT INTO facts (cik, concept, period, unit, value, accession)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (cik, concept, period, unit) DO UPDATE
  SET value     = EXCLUDED.value,
      accession = EXCLUDED.accession
"""


def upsert_facts(pool: ConnectionPool, facts: list[Fact]) -> int:
    if not facts:
        return 0
    with pool.connection() as conn, conn.cursor() as cur:
        cur.executemany(
            _UPSERT,
            [(f.cik, f.concept, f.period, f.unit, f.value, f.accession) for f in facts],
        )
        conn.commit()
    return len(facts)


def count_facts_for_cik(pool: ConnectionPool, cik: str) -> int:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM facts WHERE cik = %s", (cik,))
        row = cur.fetchone()
    return int(row[0]) if row else 0
