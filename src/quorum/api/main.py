from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from quorum.graph.build import initial_state


class CompareRequest(BaseModel):
    question: str = Field(min_length=1)
    max_replans: int = Field(default=2, ge=0, le=5)


class HealthResponse(BaseModel):
    ok: bool
    version: str = "0.1.0"


def _serialize_citations(cits: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in cits or []:
        if hasattr(c, "model_dump"):
            out.append(c.model_dump(mode="json"))
        elif isinstance(c, dict):
            out.append(c)
    return out


def _state_value(values: Any, key: str, default: Any = None) -> Any:
    if isinstance(values, dict):
        return values.get(key, default)
    return getattr(values, key, default)


def _summarize_node(node_name: str, update: Any) -> dict[str, Any]:
    # `update` is the node's returned partial-state dict from LangGraph's
    # "updates" stream mode. The analyze_axis fan-out can batch several branch
    # returns into a list, so normalize to a list and read the relevant fields.
    items = update if isinstance(update, list) else [update]
    last = items[-1] if items else {}

    if node_name == "classify":
        return {
            "axes": _state_value(last, "axes", []),
            "companies_raw": _state_value(last, "companies_raw", []),
            "out_of_scope": _state_value(last, "out_of_scope", False),
        }
    if node_name == "resolve":
        return {"tickers": _state_value(last, "tickers", [])}
    if node_name == "plan":
        tasks = _state_value(last, "plan", []) or []
        return {
            "tasks": [t.axis for t in tasks],
            "remaining_steps": _state_value(last, "remaining_steps"),
            "replan_count": _state_value(last, "replan_count"),
        }
    if node_name == "analyze_axis":
        results: list[dict[str, Any]] = []
        for it in items:
            for r in _state_value(it, "axis_results", []) or []:
                results.append(
                    {
                        "axis": r.axis,
                        "grounding": r.grounding,
                        "citations": len(r.citations),
                        "error_kind": r.error_kind,
                    }
                )
        return {"results": results}
    if node_name == "assess":
        # The route decision never survives into the update (LangGraph strips
        # unknown keys); summarize grounding instead - the next node event
        # shows the route taken.
        results = _state_value(last, "axis_results", []) or []
        return {
            "axes": len(results),
            "weak": sum(1 for r in results if getattr(r, "grounding", "") == "weak"),
        }
    if node_name == "critic":
        c = _state_value(last, "critique")
        if c is None:
            return {"critique": None}
        return {
            "status": c.status,
            "turns_used": c.turns_used,
            "duration_ms": c.duration_ms,
            "tool_calls": [
                {"tool": tc.tool, "args": tc.args, "ok": tc.ok, "result": tc.result_summary}
                for tc in c.tool_calls
            ],
            "flags": [
                {"axis": f.source_axis, "flag": f.flag, "claim": f.claim, "reason": f.reason}
                for f in c.flagged_claims
            ],
        }
    if node_name == "synthesize":
        return {
            "status": _state_value(last, "status"),
            "citations": len(_state_value(last, "report_citations", []) or []),
        }
    if node_name == "refuse":
        return {"status": _state_value(last, "status", "refused")}
    return {}


# The api factory pattern. Tests build an app with stubs injected via the
# `compiled_graph` / `ready_check` params; the production entrypoint composes
# the real dependencies in `production_lifespan` and stores them on app.state.
def create_app(
    *,
    compiled_graph: Any | None = None,
    ready_check: Callable[[], Awaitable[dict[str, Any]]] | None = None,
    lifespan: Any | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def _default_lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="Quorum", lifespan=lifespan or _default_lifespan)

    def _graph(request: Request) -> Any | None:
        if compiled_graph is not None:
            return compiled_graph
        return getattr(request.app.state, "graph", None)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(ok=True)

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, Any]:
        rc = (
            ready_check
            if ready_check is not None
            else getattr(request.app.state, "ready_check", None)
        )
        if rc is None:
            return {"ok": True, "checks": {}}
        result = await rc()
        if not result.get("ok", False):
            raise HTTPException(status_code=503, detail=result)
        return result

    @app.post("/compare")
    async def compare(req: CompareRequest, request: Request) -> EventSourceResponse:
        graph = _graph(request)
        if graph is None:
            raise HTTPException(status_code=503, detail="graph not configured")

        async def event_stream() -> AsyncIterator[dict[str, Any]]:
            state = initial_state(req.question, max_replans=req.max_replans)
            # thread_id is required once a checkpointer is attached; reusing the
            # request_id makes the run resumable at /runs/{request_id}/resume.
            config = {"configurable": {"thread_id": state.request_id}}
            try:
                async for chunk in graph.astream(state, config=config):
                    for node_name, update in (chunk or {}).items():
                        try:
                            detail = _summarize_node(node_name, update)
                        except Exception:  # noqa: BLE001
                            detail = {}
                        yield {
                            "event": "node",
                            "data": json.dumps({"node": node_name, "detail": detail}),
                        }
                # Read the final state from the checkpointer instead of running
                # the graph a second time.
                snapshot = await graph.aget_state(config)
                values = snapshot.values
                yield {
                    "event": "final",
                    "data": json.dumps(
                        {
                            "request_id": state.request_id,
                            "report": _state_value(values, "report"),
                            "status": _state_value(values, "status", "ok"),
                            "citations": _serialize_citations(
                                _state_value(values, "report_citations")
                            ),
                        }
                    ),
                }
            except Exception as e:  # noqa: BLE001
                yield {
                    "event": "error",
                    "data": json.dumps({"error": type(e).__name__, "detail": str(e)}),
                }

        return EventSourceResponse(event_stream())

    @app.get("/runs/{request_id}/resume")
    async def resume(request_id: str, request: Request) -> dict[str, Any]:
        graph = _graph(request)
        if graph is None:
            raise HTTPException(status_code=503, detail="graph not configured")
        # The caller supplies the original request_id, which is the checkpointer
        # thread_id. ainvoke(None, config) resumes from the last checkpoint.
        config = {"configurable": {"thread_id": request_id}}
        try:
            await graph.ainvoke(None, config=config)
            snapshot = await graph.aget_state(config)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=404, detail=f"resume failed: {type(e).__name__}: {e}"
            ) from e
        values = snapshot.values
        return {
            "request_id": request_id,
            "status": _state_value(values, "status", "ok"),
            "report": _state_value(values, "report"),
            "citations": _serialize_citations(_state_value(values, "report_citations")),
        }

    return app


def _build_embed_query(
    embedder: Any,
) -> Callable[[str], tuple[list[float], dict[str, float]]]:
    def embed_query(text: str) -> tuple[list[float], dict[str, float]]:
        out = embedder.embed([text])
        return out["dense_vecs"][0].tolist(), out["lexical_weights"][0]

    return embed_query


def _make_ready_check(pool: Any, qdrant: Any) -> Callable[[], Awaitable[dict[str, Any]]]:
    async def ready_check() -> dict[str, Any]:
        def _pg() -> bool:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone() is not None

        def _qd() -> bool:
            qdrant.get_collections()
            return True

        checks: dict[str, bool] = {}
        for name, fn in (("postgres", _pg), ("qdrant", _qd)):
            try:
                checks[name] = await asyncio.to_thread(fn)
            except Exception:  # noqa: BLE001
                checks[name] = False
        return {"ok": all(checks.values()), "checks": checks}

    return ready_check


@asynccontextmanager
async def production_lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Heavy deps (model load, DB connections) are imported and constructed here,
    # not at module import, so unit tests that import `create_app` stay light.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from qdrant_client import QdrantClient

    from quorum.cache.llm_cache import open_llm_cache
    from quorum.config.settings import get_settings
    from quorum.graph.build import build_graph
    from quorum.models.embed import BGEM3Embedder
    from quorum.models.router import get_client
    from quorum.state import CHECKPOINT_MODELS
    from quorum.trace.writer import TraceWriter, open_pool

    settings = get_settings()
    pool = open_pool(
        conninfo=settings.postgres_url,
        min_size=settings.pg_pool_min,
        max_size=settings.pg_pool_max,
    )
    qdrant = QdrantClient(url=settings.qdrant_url)
    embedder = BGEM3Embedder(device="cpu")
    embed_query = _build_embed_query(embedder)
    classifier_client = get_client("classifier", vllm_url=settings.vllm_url)
    sonnet_client = get_client("analyst")
    llm_cache = open_llm_cache(settings.cache_dir / "llm")

    # Explicit msgpack allowlist of our state models: clears langgraph's
    # "deserializing unregistered type" deprecation and pins checkpoint
    # deserialization to known types.
    serde = JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINT_MODELS))
    # Single async connection for the checkpointer is adequate for the v1
    # prototype's request volume; switch to AsyncConnectionPool if concurrent
    # /compare load grows.
    async with AsyncPostgresSaver.from_conn_string(settings.postgres_url, serde=serde) as saver:
        await saver.setup()
        app.state.graph = build_graph(
            classifier_client=classifier_client,
            sonnet_client=sonnet_client,
            pool=pool,
            qdrant=qdrant,
            embed_query=embed_query,
            checkpointer=saver,
            llm_cache=llm_cache,
            trace=TraceWriter(pool),
        )
        app.state.ready_check = _make_ready_check(pool, qdrant)
        try:
            yield
        finally:
            pool.close()


# Production entrypoint. uvicorn imports `app` from this module.
app = create_app(lifespan=production_lifespan)
