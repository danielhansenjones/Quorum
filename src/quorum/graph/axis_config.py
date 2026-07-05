from __future__ import annotations

from typing import Literal

# Locked taxonomy (decisions.md #1). The classifier output enum must match
# this exactly.
SUPPORTED_AXES: tuple[str, ...] = ("profitability", "growth", "leverage", "risk_factors")
AxisName = Literal["profitability", "growth", "leverage", "risk_factors"]

# Axis -> mode mapping. Quant axes drive structured retrieval via XBRL facts;
# qual axes drive semantic retrieval via Qdrant.
AXIS_MODE: dict[str, Literal["structured", "semantic"]] = {
    "profitability": "structured",
    "growth": "structured",
    "leverage": "structured",
    "risk_factors": "semantic",
}

# Normalized concept keys per quant axis. The planner pre-resolves these so the
# analyst doesn't choose; the analyst just summarizes the retrieved evidence.
AXIS_CONCEPTS: dict[str, tuple[str, ...]] = {
    "profitability": (
        "profitability.revenue",
        "profitability.net_income",
        "profitability.operating_income",
        "profitability.gross_profit",
    ),
    "growth": ("profitability.revenue",),
    "leverage": (
        "leverage.long_term_debt",
        "leverage.current_debt",
        "leverage.short_term_debt",
        "leverage.total_equity",
        "leverage.cash_and_equivalents",
    ),
}

# Semantic-axis routing: section filter + query template.
AXIS_SEMANTIC: dict[str, dict[str, list[str] | str]] = {
    "risk_factors": {
        "sections": ["item_1a_risk_factors"],
        "query": "primary business risks competitive threats supply chain regulation",
    },
}


def axis_query_or_concept(axis: str) -> str:
    if AXIS_MODE.get(axis) == "structured":
        return ",".join(AXIS_CONCEPTS.get(axis, ()))
    sem = AXIS_SEMANTIC.get(axis, {})
    return str(sem.get("query", axis))
