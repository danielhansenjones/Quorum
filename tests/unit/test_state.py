from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from quorum.state import (
    AxisResult,
    AxisTask,
    CompanyAxisFinding,
    QualCitation,
    QuantCitation,
    QuorumState,
    reduce_axis_results,
    reduce_plan,
)
from quorum.state.citation import Citation


def _q_citation() -> QuantCitation:
    return QuantCitation(
        claim="AAPL revenue is $383B",
        ticker="AAPL",
        accession="0000320193-25-000079",
        concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        normalized="profitability.revenue",
        value="383285000000",
        period="FY2025",
        unit="USD",
    )


def _ax_result(axis: str, attempts: int = 1) -> AxisResult:
    return AxisResult(
        axis=axis,
        mode="structured",
        per_company={
            "AAPL": CompanyAxisFinding(ticker="AAPL", values={}, passages=[], citations=[]),
        },
        comparison=f"axis={axis}",
        citations=[_q_citation()],
        grounding="ok",
        attempts=attempts,
    )


def test_quant_citation_kind_locked() -> None:
    c = _q_citation()
    assert c.kind == "quant"


def test_qual_citation_kind_locked() -> None:
    c = QualCitation(
        claim="x",
        ticker="AAPL",
        accession="0000320193-25-000079",
        section="item_1a_risk_factors",
        chunk_id="0000320193-25-000079:item_1a_risk_factors:0003",
        quote="supply chain disruption",
    )
    assert c.kind == "qual"


def test_citation_discriminator_picks_quant() -> None:
    adapter = TypeAdapter(Citation)
    obj = {
        "kind": "quant",
        "claim": "x",
        "ticker": "AAPL",
        "accession": "0000320193-25-000079",
        "concept": "us-gaap:NetIncomeLoss",
        "value": "93000000000",
        "period": "FY2025",
        "unit": "USD",
    }
    parsed = adapter.validate_python(obj)
    assert isinstance(parsed, QuantCitation)


def test_citation_discriminator_rejects_unknown_kind() -> None:
    adapter = TypeAdapter(Citation)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "bogus", "claim": "x"})


def test_reduce_axis_results_upsert_on_concurrent_writes() -> None:
    # Two parallel branches writing for axis="profitability" must collapse.
    left = [_ax_result("profitability", attempts=1)]
    right = [_ax_result("profitability", attempts=2)]
    merged = reduce_axis_results(left, right)
    assert len(merged) == 1
    assert merged[0].attempts == 2  # last write wins


def test_reduce_axis_results_keeps_distinct_axes() -> None:
    left = [_ax_result("profitability")]
    right = [_ax_result("leverage")]
    merged = reduce_axis_results(left, right)
    axes = sorted(r.axis for r in merged)
    assert axes == ["leverage", "profitability"]


def test_reduce_plan_overwrite_by_axis() -> None:
    t1 = AxisTask(
        axis="profitability",
        mode="structured",
        tickers=["AAPL"],
        query_or_concept="profitability.revenue",
    )
    t2 = AxisTask(
        axis="profitability",
        mode="semantic",  # re-plan changed the mode
        tickers=["AAPL"],
        query_or_concept="apple revenue trend",
    )
    merged = reduce_plan([t1], [t2])
    assert len(merged) == 1
    assert merged[0].mode == "semantic"


def test_quorum_state_serializes_round_trip() -> None:
    # Phase 5 gate: state must round-trip through JSON (LangGraph + Postgres
    # checkpointer round-trip is integration; this is the unit-level proxy).
    now = datetime.now(UTC)
    s = QuorumState(
        request_id=str(uuid4()),
        trace_id=str(uuid4()),
        request_started_at=now,
        request_deadline=now + timedelta(seconds=180),
        question="Compare AAPL and MSFT on profitability",
        tickers=["AAPL", "MSFT"],
        axes=["profitability"],
        plan=[
            AxisTask(
                axis="profitability",
                mode="structured",
                tickers=["AAPL", "MSFT"],
                query_or_concept="profitability.revenue",
            )
        ],
        axis_results=[_ax_result("profitability")],
    )
    encoded = s.model_dump_json()
    decoded = QuorumState.model_validate_json(encoded)
    assert decoded.request_id == s.request_id
    assert len(decoded.axis_results) == 1
    assert decoded.axis_results[0].citations[0].kind == "quant"
    # Reducers are typing-level metadata; round-trip preserves the list contents.
    assert [t.axis for t in decoded.plan] == ["profitability"]
