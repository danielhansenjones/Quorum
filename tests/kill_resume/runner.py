from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

from diskcache import Cache
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import quorum.graph.build as build_mod
import quorum.graph.nodes.analyze_axis as analyze_axis_mod
import quorum.graph.nodes.critic as critic_mod
import quorum.graph.nodes.synthesize as synthesize_mod
from quorum.state import CHECKPOINT_MODELS
from quorum.state.quorum_state import QuorumState
from quorum.trace.writer import TraceWriter, open_pool
from tests.kill_resume.fakes import (
    FakeClassifier,
    FakeQdrant,
    FakeSonnet,
    _maybe_hook,
    embed_query_stub,
    parse_hook_spec,
    route_of,
)

DEFAULT_QUESTION = "Compare Apple and Microsoft on profitability, growth and risk factors."


class KRMigratedState(QuorumState):
    # T8 stand-in for "a deploy added an optional state field". Module-level so
    # the checkpoint serde can resolve it by reference across processes.
    kr_migration_probe: str = "default"


# "plan" and "analyze_axis" nodes each dispatch to two module-level functions;
# both must fire the same hook point so "before:plan" pauses whichever path the
# node takes.
_BEFORE_NODE_TARGETS: dict[str, tuple[str, ...]] = {
    "classify": ("classify",),
    "resolve": ("resolve",),
    "plan": ("initial_plan", "revise_plan"),
    "analyze_axis": ("analyze_axis", "analyze_axis_agentic"),
    "assess": ("assess",),
    "critic": ("critic",),
    "synthesize": ("synthesize",),
    "rebut": ("rebut",),
    "refuse": ("refuse",),
}


def _wrap_before(attr: str, point: str) -> None:
    # build.py's node closures resolve these names as build-module globals at
    # call time, so patching the module attribute is enough.
    fn = getattr(build_mod, attr)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        _maybe_hook(point)
        return fn(*args, **kwargs)

    setattr(build_mod, attr, wrapped)


def _wrap_cached_chat(mod: ModuleType) -> None:
    # Fires BEFORE delegating so the pause happens even on what would be a
    # disk-cache hit; "kill mid LLM call" must not depend on cache state.
    fn: Callable[..., Any] = mod.chat_maybe_cached

    def wrapped(client: Any, cache: Any, **kwargs: Any) -> Any:
        _maybe_hook(f"in_cached_chat:{route_of(kwargs['messages'])}")
        return fn(client, cache, **kwargs)

    mod.chat_maybe_cached = wrapped


def _install_hooks(spec: str) -> None:
    point, _ = parse_hook_spec(spec)
    if point.startswith("before:"):
        node = point.removeprefix("before:")
        for attr in _BEFORE_NODE_TARGETS[node]:
            _wrap_before(attr, f"before:{node}")
        return
    if point.startswith("in_cached_chat:"):
        for mod in (analyze_axis_mod, critic_mod, synthesize_mod):
            _wrap_cached_chat(mod)
        return
    raise ValueError(f"unknown hook spec: {spec!r}")


def _checkpoint_summaries(history: list[Any]) -> list[dict[str, Any]]:
    # langgraph 1.x dropped metadata["writes"]; the equivalent signal is each
    # snapshot's tasks: a task with a non-None result completed, and its writes
    # are what the NEXT checkpoint applied. So checkpoint i's "writes" are the
    # completed task names of checkpoint i-1 (oldest-first).
    out: list[dict[str, Any]] = []
    for i, snap in enumerate(history):
        if i == 0:
            writes: list[str] = []
        else:
            writes = sorted({t.name for t in history[i - 1].tasks if t.result is not None})
        meta = snap.metadata or {}
        out.append(
            {
                "step": meta.get("step"),
                "source": meta.get("source"),
                "writes": writes,
                "next": list(snap.next),
            }
        )
    return out


async def _amain() -> int:
    db_url = os.environ["KR_POSTGRES_URL"]
    cache_dir = os.environ["KR_CACHE_DIR"]
    out_path = Path(os.environ["KR_OUT"])
    request_id = os.environ["KR_REQUEST_ID"]
    trace_id = os.environ["KR_TRACE_ID"]
    mode = os.environ["KR_MODE"]
    question = os.environ.get("KR_QUESTION", DEFAULT_QUESTION)
    dump = os.environ.get("KR_DUMP_CHECKPOINTS") == "1"
    migrated_schema = os.environ.get("KR_EXTRA_STATE_FIELD") == "1"

    allowed = list(CHECKPOINT_MODELS)
    if migrated_schema:
        # build_graph resolves QuorumState via its own module global (build.py
        # does `from quorum.state.quorum_state import QuorumState`), so the
        # patch must land on quorum.graph.build, not the state module.
        build_mod.QuorumState = KRMigratedState
        allowed.append(KRMigratedState)

    pool = open_pool(conninfo=db_url, min_size=2, max_size=8)
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=allowed)
        async with AsyncPostgresSaver.from_conn_string(db_url, serde=serde) as saver:
            await saver.setup()
            graph = build_mod.build_graph(
                classifier_client=FakeClassifier(),
                sonnet_client=FakeSonnet(),
                pool=pool,
                qdrant=FakeQdrant(),
                embed_query=embed_query_stub,
                checkpointer=saver,
                llm_cache=Cache(cache_dir),
                trace=TraceWriter(pool),
                critic_enabled=True,
            )
            config = {"configurable": {"thread_id": request_id}}
            # durability="sync" persists each checkpoint before the next
            # superstep starts. The default async mode leaves a flush race
            # between "node entered" and "prior checkpoint committed" that
            # would make the SIGKILL boundary nondeterministic.
            if mode == "start":
                now = datetime.now(UTC)
                state = QuorumState(
                    request_id=request_id,
                    trace_id=trace_id,
                    request_started_at=now,
                    request_deadline=now + timedelta(seconds=180),
                    question=question,
                    max_replans=2,
                )
                await graph.ainvoke(state, config=config, durability="sync")
            else:
                await graph.ainvoke(None, config=config, durability="sync")

            snap = await graph.aget_state(config)
            values = snap.values
            axis_results = values.get("axis_results", [])
            out: dict[str, Any] = {
                "status": values.get("status"),
                "report": values.get("report"),
                "axes": sorted(r.axis for r in axis_results),
                "groundings": {r.axis: r.grounding for r in axis_results},
                "replan_count": values.get("replan_count"),
                "remaining_steps": values.get("remaining_steps"),
                "rebuttal_rounds": values.get("rebuttal_rounds"),
                "n_citations": len(values.get("report_citations", [])),
            }
            if migrated_schema:
                # "absent" is a sentinel: it lets the test distinguish a channel
                # langgraph never materialized from one filled with the schema
                # default.
                out["kr_migration_probe"] = values.get("kr_migration_probe", "absent")
            if dump:
                history = [s async for s in graph.aget_state_history(config)]
                history.reverse()
                out["checkpoints"] = _checkpoint_summaries(history)
            out_path.write_text(json.dumps(out))
    finally:
        pool.close()
    return 0


def main() -> int:
    hook = os.environ.get("KR_HOOK")
    if hook:
        _install_hooks(hook)
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
