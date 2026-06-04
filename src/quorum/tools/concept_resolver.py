from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from psycopg_pool import ConnectionPool

from quorum.config.companies import CIK_BY_TICKER
from quorum.ingest.aliases import DEFAULT_TICKER_TOKEN


@dataclass(frozen=True, slots=True)
class ResolvedFact:
    value: float
    unit: str
    period: str
    accession: str
    resolved_concept: str


def _looks_like_normalized_key(key: str) -> bool:
    # Normalized keys: lower.snake.case with a dot, no colon (e.g. profitability.revenue).
    # Raw XBRL: us-gaap:Revenues style (colon, mixed case after colon).
    return "." in key and ":" not in key


def _alias_chain(pool: ConnectionPool, *, axis_metric_key: str, ticker: str) -> list[str]:
    # Try ticker-specific chain first; if empty, fall through to default chain.
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT concept FROM concept_aliases "
            "WHERE axis_metric_key = %s AND ticker_or_default = %s "
            "ORDER BY ordering",
            (axis_metric_key, ticker),
        )
        per_ticker = [row[0] for row in cur.fetchall()]
        if per_ticker:
            return per_ticker
        cur.execute(
            "SELECT concept FROM concept_aliases "
            "WHERE axis_metric_key = %s AND ticker_or_default = %s "
            "ORDER BY ordering",
            (axis_metric_key, DEFAULT_TICKER_TOKEN),
        )
        return [row[0] for row in cur.fetchall()]


def _query_facts(
    pool: ConnectionPool, *, cik: str, concepts: list[str], periods: list[str] | None
) -> dict[tuple[str, str], dict[str, Any]]:
    # Returns: {(period, concept) -> {value, unit, accession}}
    if not concepts:
        return {}
    where_periods = ""
    args: list[Any] = [cik, concepts]
    if periods:
        where_periods = "AND period = ANY(%s)"
        args.append(periods)
    sql = (
        "SELECT period, concept, value, unit, accession FROM facts "
        f"WHERE cik = %s AND concept = ANY(%s) {where_periods}"
    )
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        for period, concept, value, unit, accession in cur.fetchall():
            out[(period, concept)] = {
                "value": float(value) if value is not None else None,
                "unit": unit,
                "accession": accession,
            }
    return out


def get_financial_concept(
    pool: ConnectionPool,
    *,
    ticker: str,
    key: str,
    periods: list[str] | None = None,
) -> list[ResolvedFact]:
    # Tool contract (Phase 4b, ARCHITECTURE):
    # - Accepts raw XBRL concept (escape hatch) or normalized key (production path).
    # - Returns one row per period: first non-null concept in the fallback chain wins.
    # - Empty list on unknown ticker or no facts; never raises.
    cik = CIK_BY_TICKER.get(ticker)
    if cik is None:
        return []

    if _looks_like_normalized_key(key):
        concepts = _alias_chain(pool, axis_metric_key=key, ticker=ticker)
    else:
        concepts = [key]

    if not concepts:
        return []

    matched = _query_facts(pool, cik=cik, concepts=concepts, periods=periods)
    if not matched:
        return []

    # Pick first non-null concept per period, following the chain ordering.
    period_set = {p for (p, _) in matched}
    out: list[ResolvedFact] = []
    for period in sorted(period_set):
        for concept in concepts:
            row = matched.get((period, concept))
            if row and row["value"] is not None:
                out.append(
                    ResolvedFact(
                        value=row["value"],
                        unit=row["unit"],
                        period=period,
                        accession=row["accession"],
                        resolved_concept=concept,
                    )
                )
                break
    return out
