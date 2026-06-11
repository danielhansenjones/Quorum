from __future__ import annotations

from typing import Any

from quorum.config.companies import CIK_BY_TICKER
from quorum.state.critique import ToolCallRecord


def _ticker_issue(t: Any) -> str | None:
    if not t:
        return "missing ticker"
    if t not in CIK_BY_TICKER:
        return f"ticker not in corpus: {t}"
    return None


def validate_tool_call(rec: ToolCallRecord) -> list[str]:
    # Phase 10i: a recorded critic tool call is well-formed if its args name an
    # in-corpus ticker, a non-empty concept/query, and (for section reads) an
    # accession + section. Deterministic, no IO.
    issues: list[str] = []
    args = rec.args
    if rec.tool == "search_filings":
        q = args.get("query")
        if not (isinstance(q, str) and q.strip()):
            issues.append("empty query")
        for t in args.get("tickers") or []:
            if t not in CIK_BY_TICKER:
                issues.append(f"ticker not in corpus: {t}")
    elif rec.tool == "get_financial_concept":
        if (ti := _ticker_issue(args.get("ticker"))) is not None:
            issues.append(ti)
        key = args.get("key")
        if not (isinstance(key, str) and key.strip()):
            issues.append("empty concept key")
    elif rec.tool == "get_filing_section":
        if (ti := _ticker_issue(args.get("ticker"))) is not None:
            issues.append(ti)
        if not str(args.get("accession") or "").strip():
            issues.append("missing accession")
        if not str(args.get("section") or "").strip():
            issues.append("missing section")
    else:
        issues.append(f"unknown tool: {rec.tool}")
    return issues


def validate_tool_calls(records: list[ToolCallRecord]) -> dict[str, Any]:
    per_call = [{"tool": r.tool, "ok": r.ok, "issues": validate_tool_call(r)} for r in records]
    n = len(per_call)
    clean = sum(1 for c in per_call if not c["issues"])
    return {
        "n": n,
        "valid": clean,
        "valid_fraction": clean / n if n else 1.0,
        "calls": per_call,
    }
