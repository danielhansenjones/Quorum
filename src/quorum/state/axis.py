from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from quorum.state.citation import Citation

Mode = Literal["structured", "semantic"]
Grounding = Literal["ok", "weak", "insufficient"]
ErrorKind = Literal["none", "transient", "terminal"]


class AxisTask(BaseModel):
    model_config = ConfigDict(frozen=True)
    axis: str
    mode: Mode
    tickers: list[str]
    query_or_concept: str
    sections: list[str] | None = None


class CompanyAxisFinding(BaseModel):
    model_config = ConfigDict(frozen=True)
    ticker: str
    # Model-generated per-company narrative for this axis ("how is it doing").
    # Numbers live in values/citations; this is the qualitative read.
    assessment: str = ""
    # metric/period -> value as text; preserves units and notation as-cited.
    values: dict[str, str] = {}
    passages: list[str] = []
    citations: list[Citation] = []


class AxisResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    axis: str
    mode: Mode
    per_company: dict[str, CompanyAxisFinding]
    comparison: str
    citations: list[Citation] = []
    grounding: Grounding
    attempts: int = 1
    error_kind: ErrorKind = "none"
    error_reason: str | None = None
