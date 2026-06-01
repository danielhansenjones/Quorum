from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from quorum.state.axis import AxisResult, AxisTask
from quorum.state.citation import Citation
from quorum.state.critique import Critique, Rebuttal
from quorum.state.reducers import reduce_axis_results, reduce_plan

Status = Literal["pending", "ok", "refused", "partial"]


class QuorumState(BaseModel):
    # The Postgres LangGraph checkpointer serializes this object at every node
    # transition. Reducers live on the fields that have parallel writers
    # (axis_results, plan); everything else is last-write-wins under LangGraph's
    # default reducer.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    trace_id: str
    request_started_at: datetime
    request_deadline: datetime

    question: str
    companies_raw: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    axes: list[str] = Field(default_factory=list)
    out_of_scope: bool = False
    refusal_reason: str | None = None

    plan: Annotated[list[AxisTask], reduce_plan] = Field(default_factory=list)
    remaining_steps: int = 0
    replan_count: int = 0

    axis_results: Annotated[list[AxisResult], reduce_axis_results] = Field(default_factory=list)

    critique: Critique | None = None
    # Phase 13a. Analyst responses to critic flags; single-writer (rebut node),
    # so last-write-wins. Empty unless rebuttal_enabled and the critic flagged.
    rebuttals: list[Rebuttal] = Field(default_factory=list)

    report: str | None = None
    report_citations: list[Citation] = Field(default_factory=list)
    status: Status = "pending"
