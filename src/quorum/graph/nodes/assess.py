from __future__ import annotations

from datetime import datetime
from typing import Literal

from quorum.state.axis import AxisResult

Route = Literal["plan", "critic", "synthesize"]


def assess(
    *,
    axis_results: list[AxisResult],
    remaining_steps: int,
    request_deadline: datetime,
    now: datetime,
    replan_count: int = 0,
    max_replans: int = 2,
) -> dict[str, Route | list[AxisResult]]:
    # Phase 6e: pure function returning the next-node label plus axis_results.
    # A `weak` axis is a complete, usable analysis (citations + caveated prose);
    # it is preserved and rendered with a caveat downstream, never discarded.
    # Only the analyst marks an axis `insufficient` (empty evidence / hard
    # failure). assess re-plans weak axes while budget remains and the caller's
    # replan cap is not spent (remaining_steps stays the runaway safety net).
    if now >= request_deadline:
        return {"_route": "synthesize", "axis_results": list(axis_results)}

    has_weak = any(r.grounding == "weak" for r in axis_results)
    if has_weak and remaining_steps > 0 and replan_count < max_replans:
        return {"_route": "plan", "axis_results": list(axis_results)}
    return {"_route": "critic", "axis_results": list(axis_results)}
