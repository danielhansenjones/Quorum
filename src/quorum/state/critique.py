from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from quorum.state.citation import Citation

CritiqueStatus = Literal["ok", "timeout", "failed", "partial"]
Groundedness = Literal["ok", "thin", "unsupported"]
FlagKind = Literal["unsupported", "weakly_supported", "contradicted"]
RebuttalDisposition = Literal["defended", "retracted", "revised"]


class AxisAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)
    axis: str
    groundedness: Groundedness
    notes: str
    missed_evidence: list[str] = []


class FlaggedClaim(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_axis: str
    claim: str
    flag: FlagKind
    reason: str
    counter_citation: Citation | None = None


class Rebuttal(BaseModel):
    # Phase 13a. The analyst's response to one critic-flagged claim: defend it
    # with a citation, retract it, or revise it. The exchange outcome synthesize
    # adjudicates against the original claim and the critic's flag.
    model_config = ConfigDict(frozen=True)
    source_axis: str
    claim: str
    disposition: RebuttalDisposition
    reason: str
    citation: Citation | None = None


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool: str
    args: dict[str, Any]
    ok: bool
    result_summary: str


class Critique(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: CritiqueStatus
    per_axis: dict[str, AxisAssessment]
    cross_axis: list[str] = []
    flagged_claims: list[FlaggedClaim] = []
    tool_calls: list[ToolCallRecord] = []
    turns_used: int = 0
    duration_ms: int = 0
