from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from diskcache import Cache
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient

from quorum.graph.nodes.analyze_axis import analyze_axis, analyze_axis_agentic
from quorum.graph.nodes.assess import assess
from quorum.graph.nodes.classify import classify
from quorum.graph.nodes.critic import critic
from quorum.graph.nodes.plan import initial_plan, revise_plan
from quorum.graph.nodes.rebut import rebut
from quorum.graph.nodes.refuse import refuse
from quorum.graph.nodes.resolve import resolve
from quorum.graph.nodes.synthesize import synthesize
from quorum.models.router import ChatClient
from quorum.state.axis import AxisResult, AxisTask
from quorum.state.critique import Critique
from quorum.state.quorum_state import QuorumState
from quorum.trace.writer import TraceCtx, TraceWriter

DEFAULT_REQUEST_DEADLINE_S = 180.0
DEFAULT_MAX_REPLANS = 2


def replan_targets(plan: list[AxisTask], axis_results: list[AxisResult]) -> list[AxisTask]:
    # Pure fan-out filter. An axis whose result is already grounded (or
    # terminally insufficient) keeps that result; re-running it on a re-plan
    # pass re-bills the analyst with no new information. Only weak axes - the
    # ones revise_plan rebuilt - go back out. First pass has no results yet,
    # so everything runs.
    done = {r.axis for r in axis_results if r.grounding != "weak"}
    return [t for t in plan if t.axis not in done]


def route_after_critic(
    critique: Critique | None, remaining_steps: int, rebuttal_rounds: int = 0
) -> str:
    # Phase 13a routing, pure so it is unit-testable and the orchestration stays
    # deterministic. Unresolved flags + budget + no prior round -> the rebuttal
    # exchange; else synthesize. One round only: `rebuttals` is last-write-wins
    # state, so a second pass would overwrite round 1 and a claim retracted
    # there would escape the synthesizer's strip.
    if (
        critique is not None
        and critique.flagged_claims
        and remaining_steps > 0
        and rebuttal_rounds == 0
    ):
        return "rebut"
    return "synthesize"


def _ensure_request_metadata(state: QuorumState) -> dict[str, Any]:
    # Entry node responsibility (Phase 7). Populate request_id / trace_id /
    # deadlines if the caller hasn't set them. Idempotent on a resumed run.
    updates: dict[str, Any] = {}
    if not state.request_id:
        updates["request_id"] = str(uuid4())
    if not state.trace_id:
        updates["trace_id"] = str(uuid4())
    return updates


def build_graph(
    *,
    classifier_client: ChatClient,
    sonnet_client: ChatClient,
    pool: ConnectionPool,
    qdrant: QdrantClient,
    embed_query: Callable[[str], tuple[list[float], dict[str, float]]],
    checkpointer: Any | None = None,
    concurrency_cap: int = 4,
    llm_cache: Cache | None = None,
    trace: TraceWriter | None = None,
    critic_enabled: bool = True,
    rebuttal_enabled: bool = False,
    agentic_analyst: bool = False,
    legwork_client: ChatClient | None = None,
) -> Any:
    # Global semaphore caps in-flight analyze_axis branches (Phase 7 gate).
    sem = threading.Semaphore(concurrency_cap)

    def traced(
        node_name: str, fn: Callable[[QuorumState], dict[str, Any]]
    ) -> Callable[[QuorumState], dict[str, Any]]:
        def wrapped(state: QuorumState) -> dict[str, Any]:
            start = time.monotonic()
            ctx = TraceCtx(trace, state.request_id, state.trace_id)
            try:
                out = fn(state)
            except Exception as e:  # noqa: BLE001
                ctx.event(
                    node_name,
                    duration_ms=int((time.monotonic() - start) * 1000),
                    error_kind="terminal",
                    error_reason=type(e).__name__,
                )
                raise
            ctx.event(node_name, duration_ms=int((time.monotonic() - start) * 1000))
            return out

        return wrapped

    def entry_node(state: QuorumState) -> dict[str, Any]:
        # No-op if the caller already set request metadata. Otherwise mint it.
        return _ensure_request_metadata(state)

    def classify_node(state: QuorumState) -> dict[str, Any]:
        return classify(
            state.question,
            client=classifier_client,
            trace_ctx=TraceCtx(trace, state.request_id, state.trace_id),
        )

    def resolve_node(state: QuorumState) -> dict[str, Any]:
        return resolve(state.companies_raw)

    def plan_node(state: QuorumState) -> dict[str, Any]:
        if state.replan_count == 0 and not state.plan:
            return initial_plan(axes=state.axes, tickers=state.tickers)
        return revise_plan(
            plan=state.plan,
            axis_results=state.axis_results,
            remaining_steps=state.remaining_steps,
            replan_count=state.replan_count,
        )

    def analyze_axis_node(payload: dict[str, Any]) -> dict[str, Any]:
        # Fan-out arrival: payload carries one AxisTask plus the request ids the
        # Send copied from state (fan-out branches don't see the full state).
        from quorum.state.axis import AxisTask

        start = time.monotonic()
        ctx = TraceCtx(trace, payload.get("request_id"), payload.get("trace_id"))
        task = (
            payload["task"]
            if isinstance(payload["task"], AxisTask)
            else AxisTask(**payload["task"])
        )
        with sem:
            if agentic_analyst and legwork_client is not None:
                result = analyze_axis_agentic(
                    task,
                    legwork_client=legwork_client,
                    sonnet_client=sonnet_client,
                    pool=pool,
                    qdrant=qdrant,
                    embed_query=embed_query,
                    llm_cache=llm_cache,
                    trace_ctx=ctx,
                )
            else:
                result = analyze_axis(
                    task,
                    sonnet_client=sonnet_client,
                    pool=pool,
                    qdrant=qdrant,
                    embed_query=embed_query,
                    llm_cache=llm_cache,
                    trace_ctx=ctx,
                )
        ctx.event(
            f"analyze_axis:{task.axis}",
            duration_ms=int((time.monotonic() - start) * 1000),
            error_kind=result.error_kind,
            error_reason=result.error_reason,
            input_shape={"axis": task.axis, "mode": task.mode, "tickers": task.tickers},
        )
        # The axis_results reducer upserts by axis name; parallel writes collapse.
        return {"axis_results": [result]}

    def fan_out(state: QuorumState) -> list[Send]:
        if not state.plan:
            return []
        return [
            Send(
                "analyze_axis",
                {"task": task, "request_id": state.request_id, "trace_id": state.trace_id},
            )
            for task in replan_targets(state.plan, state.axis_results)
        ]

    def assess_node(state: QuorumState) -> dict[str, Any]:
        out = assess(
            axis_results=state.axis_results,
            remaining_steps=state.remaining_steps,
            request_deadline=state.request_deadline,
            now=datetime.now(UTC),
            replan_count=state.replan_count,
            max_replans=state.max_replans,
        )
        return {"axis_results": out["axis_results"]}

    def critic_node(state: QuorumState) -> dict[str, Any]:
        # Containment property: any failure -> critique=None and pass through.
        # state.rebuttals is [] unless the rebuttal loop ran; the critic prompt
        # is byte-identical when it is empty.
        try:
            c = critic(
                state.axis_results,
                sonnet_client=sonnet_client,
                pool=pool,
                qdrant=qdrant,
                embed_query=embed_query,
                llm_cache=llm_cache,
                trace_ctx=TraceCtx(trace, state.request_id, state.trace_id),
                rebuttals=state.rebuttals,
            )
        except Exception:  # noqa: BLE001
            c = None
        return {"critique": c}

    def rebut_node(state: QuorumState) -> dict[str, Any]:
        flags = state.critique.flagged_claims if state.critique else []
        out = rebut(
            flagged_claims=flags,
            axis_results=state.axis_results,
            remaining_steps=state.remaining_steps,
            sonnet_client=sonnet_client,
            llm_cache=llm_cache,
            trace_ctx=TraceCtx(trace, state.request_id, state.trace_id),
        )
        out["rebuttal_rounds"] = state.rebuttal_rounds + 1
        return out

    def synthesize_node(state: QuorumState) -> dict[str, Any]:
        return synthesize(
            axis_results=state.axis_results,
            critique=state.critique,
            sonnet_client=sonnet_client,
            question=state.question,
            rebuttals=state.rebuttals,
            llm_cache=llm_cache,
            trace_ctx=TraceCtx(trace, state.request_id, state.trace_id),
        )

    def refuse_node(state: QuorumState) -> dict[str, Any]:
        return refuse(state.refusal_reason)

    # Edges (Phase 7)
    builder: Any = StateGraph(QuorumState)
    builder.add_node("entry", traced("entry", entry_node))
    builder.add_node("classify", traced("classify", classify_node))
    builder.add_node("resolve", traced("resolve", resolve_node))
    builder.add_node("plan", traced("plan", plan_node))
    builder.add_node("analyze_axis", analyze_axis_node)  # traces itself (fan-out)
    builder.add_node("assess", traced("assess", assess_node))
    if critic_enabled:
        builder.add_node("critic", traced("critic", critic_node))
        if rebuttal_enabled:
            builder.add_node("rebut", traced("rebut", rebut_node))
    builder.add_node("synthesize", traced("synthesize", synthesize_node))
    builder.add_node("refuse", traced("refuse", refuse_node))

    builder.add_edge(START, "entry")
    builder.add_edge("entry", "classify")

    def after_classify(state: QuorumState) -> str:
        if state.out_of_scope or not state.axes:
            return "refuse"
        return "resolve"

    builder.add_conditional_edges(
        "classify",
        after_classify,
        {"resolve": "resolve", "refuse": "refuse"},
    )

    def after_resolve(state: QuorumState) -> str:
        if state.refusal_reason and len(state.tickers) < 2:
            return "refuse"
        return "plan"

    builder.add_conditional_edges(
        "resolve",
        after_resolve,
        {"plan": "plan", "refuse": "refuse"},
    )

    builder.add_conditional_edges("plan", fan_out, ["analyze_axis"])
    builder.add_edge("analyze_axis", "assess")

    def after_assess(state: QuorumState) -> str:
        # LangGraph strips keys unknown to the Pydantic schema from node
        # returns, so a "_route" hint from assess_node never survives into
        # state. The conditional edge recomputes the decision from the latest
        # state instead, keeping routing a pure function of state.
        from quorum.graph.nodes.assess import assess as assess_pure

        out = assess_pure(
            axis_results=state.axis_results,
            remaining_steps=state.remaining_steps,
            request_deadline=state.request_deadline,
            now=datetime.now(UTC),
            replan_count=state.replan_count,
            max_replans=state.max_replans,
        )
        return str(out["_route"])

    # When the critic is disabled (Phase 12a off-arm), the all-grounded "critic"
    # route short-circuits to synthesize and no critic node exists.
    builder.add_conditional_edges(
        "assess",
        after_assess,
        {
            "plan": "plan",
            "critic": "critic" if critic_enabled else "synthesize",
            "synthesize": "synthesize",
        },
    )

    def after_critic(state: QuorumState) -> str:
        return route_after_critic(state.critique, state.remaining_steps, state.rebuttal_rounds)

    if critic_enabled and rebuttal_enabled:
        builder.add_conditional_edges(
            "critic",
            after_critic,
            {"rebut": "rebut", "synthesize": "synthesize"},
        )
        builder.add_edge("rebut", "critic")
    elif critic_enabled:
        builder.add_edge("critic", "synthesize")
    builder.add_edge("synthesize", END)
    builder.add_edge("refuse", END)

    return builder.compile(checkpointer=checkpointer)


def initial_state(
    question: str,
    *,
    deadline_s: float = DEFAULT_REQUEST_DEADLINE_S,
    max_replans: int = DEFAULT_MAX_REPLANS,
) -> QuorumState:
    now = datetime.now(UTC)
    return QuorumState(
        request_id=str(uuid4()),
        trace_id=str(uuid4()),
        request_started_at=now,
        request_deadline=now + timedelta(seconds=deadline_s),
        question=question,
        max_replans=max_replans,
    )
