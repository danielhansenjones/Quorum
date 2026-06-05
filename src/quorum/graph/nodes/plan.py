from __future__ import annotations

from typing import Any

from quorum.graph.axis_config import AXIS_MODE, axis_query_or_concept
from quorum.state.axis import AxisResult, AxisTask

DEFAULT_BUDGET_MULTIPLIER = 2  # remaining_steps = 2 * num_axes (decisions.md #4)


def initial_plan(
    *,
    axes: list[str],
    tickers: list[str],
    budget_multiplier: int = DEFAULT_BUDGET_MULTIPLIER,
) -> dict[str, Any]:
    # Phase 6c first pass: deterministic expansion of (tickers x axes).
    tasks = [
        AxisTask(
            axis=axis,
            mode=AXIS_MODE.get(axis, "semantic"),
            tickers=list(tickers),
            query_or_concept=axis_query_or_concept(axis),
        )
        for axis in axes
    ]
    return {
        "plan": tasks,
        "remaining_steps": len(axes) * budget_multiplier,
        "replan_count": 0,
    }


def revise_plan(
    *,
    plan: list[AxisTask],
    axis_results: list[AxisResult],
    remaining_steps: int,
    replan_count: int,
) -> dict[str, Any]:
    # Phase 6c re-plan path: rebuild tasks ONLY for axes flagged `weak`.
    # In v1 we substitute the qualitative query template for the same axis,
    # which broadens retrieval without changing the structural mode.
    weak_axes = {r.axis for r in axis_results if r.grounding == "weak"}
    if not weak_axes or remaining_steps <= 0:
        return {}

    revised: list[AxisTask] = []
    for task in plan:
        if task.axis not in weak_axes:
            continue
        # Simple "try harder" for v1: switch to semantic and widen the query.
        revised.append(
            AxisTask(
                axis=task.axis,
                mode="semantic",
                tickers=task.tickers,
                query_or_concept=f"{task.axis} {task.query_or_concept} year over year",
                sections=None,
            )
        )
    return {
        "plan": revised,
        "remaining_steps": remaining_steps - 1,
        "replan_count": replan_count + 1,
    }
