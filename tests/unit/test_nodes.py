from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from quorum.graph.axis_config import AXIS_MODE, SUPPORTED_AXES
from quorum.graph.nodes.assess import assess
from quorum.graph.nodes.plan import initial_plan, revise_plan
from quorum.graph.nodes.refuse import refuse
from quorum.graph.nodes.resolve import resolve
from quorum.graph.nodes.synthesize import (
    _INSUFFICIENT_MARKER,
    _apply_rebuttals,
    _format_axis_result,
    synthesize,
)
from quorum.state.axis import AxisResult, AxisTask, CompanyAxisFinding
from quorum.state.citation import QuantCitation
from quorum.state.critique import Rebuttal


def _result(axis: str, grounding: str, attempts: int = 1) -> AxisResult:
    return AxisResult(
        axis=axis,
        mode=AXIS_MODE.get(axis, "semantic"),
        per_company={"AAPL": CompanyAxisFinding(ticker="AAPL")},
        comparison="prose",
        citations=[],
        grounding=grounding,  # type: ignore[arg-type]
        attempts=attempts,
    )


# ---- resolve ----


def test_resolve_two_in_corpus_routes_forward() -> None:
    out = resolve(["AAPL", "MSFT"])
    assert out["tickers"] == ["AAPL", "MSFT"]
    assert out.get("refusal_reason") is None


def test_resolve_one_in_corpus_triggers_refuse() -> None:
    out = resolve(["AAPL", "Nestle"])
    assert out["tickers"] == ["AAPL"]
    assert "Need at least two" in out["refusal_reason"]


def test_resolve_dedupes() -> None:
    out = resolve(["AAPL", "Apple Inc.", "MSFT"])
    assert out["tickers"] == ["AAPL", "MSFT"]


# ---- plan ----


def test_initial_plan_emits_one_task_per_axis() -> None:
    out = initial_plan(axes=["profitability", "leverage"], tickers=["AAPL", "MSFT"])
    plan: list[AxisTask] = out["plan"]
    assert len(plan) == 2
    assert {t.axis for t in plan} == {"profitability", "leverage"}
    assert all(t.tickers == ["AAPL", "MSFT"] for t in plan)


def test_initial_plan_sets_budget() -> None:
    out = initial_plan(axes=list(SUPPORTED_AXES), tickers=["AAPL", "MSFT"])
    # 4 axes * 2 = 8 (decisions.md #4)
    assert out["remaining_steps"] == 8
    assert out["replan_count"] == 0


def test_initial_plan_mode_per_axis() -> None:
    out = initial_plan(axes=["profitability", "risk_factors"], tickers=["AAPL", "MSFT"])
    by_axis = {t.axis: t for t in out["plan"]}
    assert by_axis["profitability"].mode == "structured"
    assert by_axis["risk_factors"].mode == "semantic"


def test_revise_plan_only_touches_weak_axes() -> None:
    base = initial_plan(axes=["profitability", "leverage"], tickers=["AAPL", "MSFT"])
    plan = base["plan"]
    results = [_result("profitability", "weak"), _result("leverage", "ok")]
    out = revise_plan(plan=plan, axis_results=results, remaining_steps=4, replan_count=0)
    revised: list[AxisTask] = out["plan"]
    # Only the weak axis is in the revised list (reducer upserts by axis).
    assert {t.axis for t in revised} == {"profitability"}
    assert out["replan_count"] == 1
    assert out["remaining_steps"] == 3


def test_revise_plan_skipped_when_no_weak() -> None:
    out = revise_plan(plan=[], axis_results=[_result("p", "ok")], remaining_steps=2, replan_count=0)
    assert out == {}


# ---- assess ----


def _now_and_deadline(seconds_until: int = 60) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    return now, now + timedelta(seconds=seconds_until)


def test_assess_all_ok_routes_to_critic() -> None:
    now, deadline = _now_and_deadline()
    out = assess(
        axis_results=[_result("p", "ok"), _result("l", "ok")],
        remaining_steps=2,
        request_deadline=deadline,
        now=now,
    )
    assert out["_route"] == "critic"


def test_assess_weak_with_budget_routes_to_plan() -> None:
    now, deadline = _now_and_deadline()
    out = assess(
        axis_results=[_result("p", "weak"), _result("l", "ok")],
        remaining_steps=2,
        request_deadline=deadline,
        now=now,
    )
    assert out["_route"] == "plan"


def test_assess_weak_no_budget_preserves_weak_and_routes_to_critic() -> None:
    now, deadline = _now_and_deadline()
    out = assess(
        axis_results=[_result("p", "weak")],
        remaining_steps=0,
        request_deadline=deadline,
        now=now,
    )
    # Budget exhausted: weak is a usable analysis, preserved (not discarded),
    # and routed to critic for comment.
    assert out["_route"] == "critic"
    results = out["axis_results"]
    assert results[0].grounding == "weak"


def test_assess_replan_cap_spent_routes_to_critic() -> None:
    # Weak axis and step budget remain, but the caller's replan cap is spent:
    # no further plan round (remaining_steps alone allowed up to 2 * num_axes).
    now, deadline = _now_and_deadline()
    out = assess(
        axis_results=[_result("p", "weak")],
        remaining_steps=2,
        request_deadline=deadline,
        now=now,
        replan_count=2,
        max_replans=2,
    )
    assert out["_route"] == "critic"


def test_assess_any_insufficient_routes_to_critic() -> None:
    now, deadline = _now_and_deadline()
    out = assess(
        axis_results=[_result("p", "ok"), _result("l", "insufficient")],
        remaining_steps=2,
        request_deadline=deadline,
        now=now,
    )
    # No re-plan on insufficient (terminal failure).
    assert out["_route"] == "critic"


def test_assess_deadline_exceeded_routes_to_synthesize() -> None:
    now = datetime.now(UTC)
    deadline = now - timedelta(seconds=1)
    out = assess(
        axis_results=[_result("p", "weak")],
        remaining_steps=2,
        request_deadline=deadline,
        now=now,
    )
    assert out["_route"] == "synthesize"
    results = out["axis_results"]
    # Deadline is a hard stop, but a completed weak axis is still rendered, not
    # discarded.
    assert results[0].grounding == "weak"


# ---- refuse ----


def test_refuse_emits_refused_status() -> None:
    out = refuse("Only one in-corpus company")
    assert out["status"] == "refused"
    assert "Only one" in out["report"]


# ---- synthesize ----


class _FakeSonnet:
    backend = "anthropic"
    model = "claude-sonnet-4-6"

    def __init__(self, response_text: str) -> None:
        self._text = response_text

    def chat(self, **kwargs: Any) -> Any:
        class _Block:
            text = self._text

        class _Resp:
            content = [_Block()]

        return _Resp()


def test_synthesize_strips_uncited_numbers() -> None:
    fake = _FakeSonnet(
        "### Profitability\n"
        "AAPL revenue grew 5% [AAPL:Q0] to $383B [AAPL:Q1].\n"
        "MSFT had margins of 38% with no citation here.\n"
    )
    out = synthesize(
        axis_results=[_result("profitability", "ok")],
        critique=None,
        sonnet_client=fake,
        question="compare",
    )
    # The cited line stays; the uncited "MSFT had margins" line is stripped.
    assert "[AAPL:Q0]" in out["report"]
    assert "MSFT had margins of 38%" not in out["report"]


def test_synthesize_insufficient_axis_marker() -> None:
    fake = _FakeSonnet(
        "### Profitability\n"
        "Discussion of AAPL revenue [AAPL:Q0].\n\n"
        "### Leverage\n*Insufficient data*\n"
    )
    out = synthesize(
        axis_results=[
            _result("profitability", "ok"),
            _result("leverage", "insufficient"),
        ],
        critique=None,
        sonnet_client=fake,
        question="compare",
    )
    assert "*Insufficient data*" in out["report"]
    assert out["status"] == "partial"


def _weak_result_with_evidence() -> AxisResult:
    return AxisResult(
        axis="profitability",
        mode="structured",
        per_company={
            "AAPL": CompanyAxisFinding(ticker="AAPL", assessment="Margins held ~47% [AAPL:Q0].")
        },
        comparison="AAPL outgrew the field [AAPL:Q0].",
        citations=[],
        grounding="weak",
        error_reason="GOOGL gross profit unavailable",
    )


def test_format_weak_axis_renders_with_caveat() -> None:
    block = _format_axis_result(_weak_result_with_evidence())
    assert "*Insufficient data*" not in block
    assert "AAPL outgrew the field" in block
    assert "_Caveat: GOOGL gross profit unavailable_" in block


def test_format_insufficient_axis_uses_marker() -> None:
    assert "*Insufficient data*" in _format_axis_result(_result("profitability", "insufficient"))


def test_format_weak_but_empty_falls_back_to_marker() -> None:
    r = AxisResult(
        axis="profitability",
        mode="structured",
        per_company={},
        comparison="",
        citations=[],
        grounding="weak",
    )
    assert _format_axis_result(r).strip().endswith(_INSUFFICIENT_MARKER)


def test_synthesize_weak_axis_marks_partial() -> None:
    fake = _FakeSonnet("### Profitability\nAAPL grew [AAPL:Q0].\n")
    out = synthesize(
        axis_results=[_weak_result_with_evidence()],
        critique=None,
        sonnet_client=fake,
        question="compare",
    )
    assert out["status"] == "partial"


def test_synthesize_marker_in_report_marks_partial_when_axes_grounded() -> None:
    # New rule: a grounded axis whose synthesized section still renders the
    # insufficient marker (a requested sub-metric with no isolable data) makes
    # the run partial, even though no axis was flagged weak/insufficient upstream.
    fake = _FakeSonnet(
        "### Profitability\nAAPL margins held [AAPL:Q0].\n\n"
        "### Inventory Turnover\n*Insufficient data*\n"
    )
    out = synthesize(
        axis_results=[_result("profitability", "ok")],
        critique=None,
        sonnet_client=fake,
        question="compare",
    )
    assert out["status"] == "partial"


# ---- 13a: rebuttal adjudication ----


def _quant_cite() -> QuantCitation:
    return QuantCitation(
        claim="x",
        ticker="AAPL",
        accession="a",
        concept="profitability.revenue",
        value="1",
        period="FY2024",
        unit="USD",
    )


def test_apply_rebuttals_drops_retracted_and_keeps_rest() -> None:
    report = "### Profitability\nAAPL revenue tripled [AAPL:Q0].\nMSFT held steady [MSFT:Q0].\n"
    reb = [
        Rebuttal(
            source_axis="profitability",
            claim="AAPL revenue tripled",
            disposition="retracted",
            reason="cannot support",
        )
    ]
    cleaned, extra = _apply_rebuttals(report, reb)
    assert "AAPL revenue tripled" not in cleaned
    assert "MSFT held steady" in cleaned
    assert extra == []


def test_apply_rebuttals_defended_contributes_citation() -> None:
    cite = _quant_cite()
    reb = [
        Rebuttal(
            source_axis="profitability",
            claim="c",
            disposition="defended",
            reason="supported",
            citation=cite,
        )
    ]
    cleaned, extra = _apply_rebuttals("report body unchanged", reb)
    assert cleaned == "report body unchanged"
    assert extra == [cite]


def test_synthesize_retracts_claim_and_adds_defended_citation() -> None:
    cite = _quant_cite()
    fake = _FakeSonnet(
        "### Profitability\nAAPL revenue tripled [AAPL:Q0].\nMSFT margins rose [MSFT:Q0].\n"
    )
    out = synthesize(
        axis_results=[_result("profitability", "ok")],
        critique=None,
        sonnet_client=fake,
        question="compare",
        rebuttals=[
            Rebuttal(
                source_axis="profitability",
                claim="AAPL revenue tripled",
                disposition="retracted",
                reason="cannot support",
            ),
            Rebuttal(
                source_axis="profitability",
                claim="MSFT margins rose",
                disposition="defended",
                reason="supported",
                citation=cite,
            ),
        ],
    )
    assert "AAPL revenue tripled" not in out["report"]
    assert "MSFT margins rose" in out["report"]
    assert cite in out["report_citations"]


def test_synthesize_empty_rebuttals_leave_report_unchanged() -> None:
    # The no-rebuttal arm must be untouched (cache + A/B baseline stability).
    fake = _FakeSonnet("### Profitability\nAAPL margins held [AAPL:Q0].\n")
    out = synthesize(
        axis_results=[_result("profitability", "ok")],
        critique=None,
        sonnet_client=fake,
        question="compare",
        rebuttals=[],
    )
    assert "AAPL margins held" in out["report"]
