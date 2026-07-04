from __future__ import annotations

from typing import Any

from quorum.graph.build import build_graph, initial_state, replan_targets, route_after_critic
from quorum.state.axis import AxisResult, AxisTask, CompanyAxisFinding
from quorum.state.critique import Critique, FlaggedClaim


class _FakeClient:
    backend = "anthropic"
    model = "claude-sonnet-4-6"

    def chat(self, **kwargs: Any) -> Any:
        class _Block:
            type = "text"
            text = "{}"

        class _Resp:
            content = [_Block()]
            stop_reason = "end_turn"

        return _Resp()


def test_graph_compiles_with_fake_deps() -> None:
    g = build_graph(
        classifier_client=_FakeClient(),
        sonnet_client=_FakeClient(),
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        embed_query=lambda q: ([0.0], {}),
        checkpointer=None,
    )
    # The compiled graph exposes a `get_graph()` describing the nodes and edges.
    schema = g.get_graph()
    node_names = {n for n in schema.nodes}
    expected = {
        "__start__",
        "entry",
        "classify",
        "resolve",
        "plan",
        "analyze_axis",
        "assess",
        "critic",
        "synthesize",
        "refuse",
        "__end__",
    }
    # All required nodes present.
    assert expected.issubset(node_names), f"missing: {expected - node_names}"


def test_initial_state_has_required_fields() -> None:
    s = initial_state("Compare AAPL and MSFT on profitability.")
    assert s.request_id
    assert s.trace_id
    assert s.question.startswith("Compare")
    assert s.request_deadline > s.request_started_at


def _compile(**overrides: Any) -> Any:
    return build_graph(
        classifier_client=_FakeClient(),
        sonnet_client=_FakeClient(),
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        embed_query=lambda q: ([0.0], {}),
        checkpointer=None,
        **overrides,
    )


def _edges(g: Any) -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in g.get_graph().edges}


def test_critic_disabled_removes_node_and_routes_assess_to_synthesize() -> None:
    g = _compile(critic_enabled=False)
    node_names = {n for n in g.get_graph().nodes}
    assert "critic" not in node_names
    edges = _edges(g)
    # The all-grounded route short-circuits straight to synthesize.
    assert ("assess", "synthesize") in edges
    assert not any(target == "critic" for _, target in edges)


def test_critic_enabled_keeps_assess_critic_synthesize_path() -> None:
    g = _compile(critic_enabled=True)
    assert "critic" in {n for n in g.get_graph().nodes}
    edges = _edges(g)
    assert ("assess", "critic") in edges
    assert ("critic", "synthesize") in edges


def _flags(n: int = 1) -> Critique:
    return Critique(
        status="ok",
        per_axis={},
        flagged_claims=[
            FlaggedClaim(source_axis="profitability", claim=f"c{i}", flag="unsupported", reason="r")
            for i in range(n)
        ],
    )


def test_route_after_critic_flags_with_budget_routes_to_rebut() -> None:
    assert route_after_critic(_flags(), 2) == "rebut"


def test_route_after_critic_no_budget_routes_to_synthesize() -> None:
    # Termination guard: the keeps-flagging case still ends once budget hits 0.
    assert route_after_critic(_flags(), 0) == "synthesize"


def test_route_after_critic_no_flags_routes_to_synthesize() -> None:
    assert route_after_critic(_flags(0), 2) == "synthesize"
    assert route_after_critic(None, 2) == "synthesize"


def test_route_after_critic_one_round_only() -> None:
    # A second rebut pass would overwrite round-1 rebuttals (last-write-wins
    # state) and let a retracted claim escape the synthesizer's strip.
    assert route_after_critic(_flags(), 2, rebuttal_rounds=1) == "synthesize"


def _task(axis: str) -> AxisTask:
    return AxisTask(axis=axis, mode="semantic", tickers=["AAPL", "MSFT"], query_or_concept="q")


def _axis_result(axis: str, grounding: str) -> AxisResult:
    return AxisResult(
        axis=axis,
        mode="semantic",
        per_company={"AAPL": CompanyAxisFinding(ticker="AAPL")},
        comparison="c",
        citations=[],
        grounding=grounding,  # type: ignore[arg-type]
        attempts=1,
    )


def test_replan_targets_first_pass_sends_everything() -> None:
    plan = [_task("profitability"), _task("growth")]
    assert replan_targets(plan, []) == plan


def test_replan_targets_resends_only_weak_axes() -> None:
    plan = [_task("profitability"), _task("growth"), _task("leverage")]
    results = [
        _axis_result("profitability", "ok"),
        _axis_result("growth", "weak"),
        _axis_result("leverage", "insufficient"),
    ]
    assert [t.axis for t in replan_targets(plan, results)] == ["growth"]


def test_rebuttal_enabled_adds_loop_edges() -> None:
    g = _compile(critic_enabled=True, rebuttal_enabled=True)
    nodes = {n for n in g.get_graph().nodes}
    edges = _edges(g)
    assert "rebut" in nodes
    assert ("critic", "rebut") in edges
    assert ("rebut", "critic") in edges
    assert ("critic", "synthesize") in edges  # the no-flags / no-budget exit


def test_rebuttal_disabled_has_no_rebut_node() -> None:
    g = _compile(critic_enabled=True, rebuttal_enabled=False)
    assert "rebut" not in {n for n in g.get_graph().nodes}
    assert ("critic", "synthesize") in _edges(g)
