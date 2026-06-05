from __future__ import annotations

from typing import Any

from quorum.tools.resolve_company import resolve_company

MIN_COMPANIES = 2


def resolve(companies_raw: list[str]) -> dict[str, Any]:
    # Phase 6b: deterministic, in-process. Refusal threshold is the design's
    # "at least two in-corpus companies" rule.
    tickers: list[str] = []
    for q in companies_raw:
        rc = resolve_company(q)
        if rc and rc.ticker not in tickers:
            tickers.append(rc.ticker)
    if len(tickers) < MIN_COMPANIES:
        return {
            "tickers": tickers,
            "out_of_scope": False,
            "refusal_reason": (
                "Need at least two in-corpus companies. "
                f"Resolved {len(tickers)} from input mentions: {companies_raw!r}"
            ),
        }
    return {"tickers": tickers, "refusal_reason": None}
