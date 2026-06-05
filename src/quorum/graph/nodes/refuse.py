from __future__ import annotations

from quorum.graph.axis_config import SUPPORTED_AXES


def refuse(refusal_reason: str | None) -> dict[str, str]:
    reason = refusal_reason or "Out of scope for the Quorum corpus."
    body = (
        f"{reason}\n\nQuorum supports comparisons across these axes: {', '.join(SUPPORTED_AXES)}."
    )
    return {
        "status": "refused",
        "refusal_reason": reason,
        "report": body,
    }
