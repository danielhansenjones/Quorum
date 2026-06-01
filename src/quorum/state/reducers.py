from __future__ import annotations

from quorum.state.axis import AxisResult, AxisTask


def reduce_axis_results(
    left: list[AxisResult] | None, right: list[AxisResult] | None
) -> list[AxisResult]:
    # Upsert-by-axis reducer (Phase 5 gate). Two parallel writes for the same
    # axis must collapse to one entry, not two. Re-plan also overwrites by axis,
    # not appends.
    by_axis: dict[str, AxisResult] = {}
    for r in left or []:
        by_axis[r.axis] = r
    for r in right or []:
        by_axis[r.axis] = r
    # Preserve a stable, deterministic order so prompts and traces compare cleanly.
    return [by_axis[axis] for axis in sorted(by_axis.keys())]


def reduce_plan(left: list[AxisTask] | None, right: list[AxisTask] | None) -> list[AxisTask]:
    by_axis: dict[str, AxisTask] = {}
    for t in left or []:
        by_axis[t.axis] = t
    for t in right or []:
        by_axis[t.axis] = t
    return [by_axis[axis] for axis in sorted(by_axis.keys())]
