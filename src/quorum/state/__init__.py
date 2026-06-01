from pydantic import BaseModel

from quorum.state.axis import (
    AxisResult,
    AxisTask,
    CompanyAxisFinding,
    ErrorKind,
    Grounding,
    Mode,
)
from quorum.state.citation import Citation, QualCitation, QuantCitation
from quorum.state.critique import (
    AxisAssessment,
    Critique,
    CritiqueStatus,
    FlaggedClaim,
    FlagKind,
    Groundedness,
    Rebuttal,
    RebuttalDisposition,
    ToolCallRecord,
)
from quorum.state.quorum_state import QuorumState, Status
from quorum.state.reducers import reduce_axis_results, reduce_plan

# Concrete Pydantic models the LangGraph Postgres checkpointer serializes into
# checkpoints. Passed as an explicit msgpack allowlist (see api.main) to clear
# langgraph's "deserializing unregistered type" deprecation and pin
# deserialization to known types. With an explicit allowlist, any UNREGISTERED
# type is BLOCKED on resume - so every concrete state model must be listed here.
# test_checkpoint_allowlist.py fails if a quorum.state model is missing.
CHECKPOINT_MODELS: tuple[type[BaseModel], ...] = (
    AxisTask,
    AxisResult,
    CompanyAxisFinding,
    QuantCitation,
    QualCitation,
    AxisAssessment,
    Critique,
    FlaggedClaim,
    Rebuttal,
    ToolCallRecord,
    QuorumState,
)

__all__ = [
    "CHECKPOINT_MODELS",
    "AxisAssessment",
    "AxisResult",
    "AxisTask",
    "Citation",
    "CompanyAxisFinding",
    "Critique",
    "CritiqueStatus",
    "ErrorKind",
    "FlaggedClaim",
    "FlagKind",
    "Grounding",
    "Groundedness",
    "Mode",
    "QualCitation",
    "QuantCitation",
    "QuorumState",
    "Rebuttal",
    "RebuttalDisposition",
    "Status",
    "ToolCallRecord",
    "reduce_axis_results",
    "reduce_plan",
]
