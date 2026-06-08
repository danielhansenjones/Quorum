from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from diskcache import Cache
from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient

from quorum.models.cached_chat import chat_maybe_cached
from quorum.models.router import ChatClient
from quorum.state.axis import AxisResult
from quorum.state.critique import (
    AxisAssessment,
    Critique,
    FlaggedClaim,
    Rebuttal,
    ToolCallRecord,
)
from quorum.tools.concept_resolver import get_financial_concept
from quorum.tools.filing_section import get_filing_section
from quorum.tools.search import hybrid_search
from quorum.trace.cost import llm_trace_fields
from quorum.trace.writer import TraceCtx

CRITIC_MAX_TURNS = 5
CRITIC_WALL_CLOCK_S = 90.0

# Anthropic tool schemas. The critic is a real agentic tool-loop: Sonnet picks
# the tool per turn from this set. Args are validated structurally and
# anything else surfaces as a tool error (not an exception).
_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "search_filings",
        "description": "Hybrid retrieval over filing chunks. Use for qualitative claims.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "tickers": {"type": "array", "items": {"type": "string"}},
                "sections": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_financial_concept",
        "description": "Resolve a normalized concept (e.g. profitability.revenue) or raw XBRL concept.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "key": {"type": "string"},
                "periods": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ticker", "key"],
        },
    },
    {
        "name": "get_filing_section",
        "description": "Return the full text of one filing section.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "accession": {"type": "string"},
                "section": {"type": "string"},
            },
            "required": ["ticker", "accession", "section"],
        },
    },
]

_CRITIC_SYSTEM = (
    "You are Quorum's critic. The Quorum analysts have already produced per-axis "
    "comparisons grounded in retrieved evidence. Your job is to VERIFY: check that "
    "each claim is supported by the cited evidence, surface evidence the analysts "
    "may have missed, and flag any unsupported, weakly-supported, or contradicted "
    "claims.\n\n"
    "You have up to 5 tool-use turns and 90 seconds. Be efficient. When done, "
    "respond with one final JSON object using stop_reason `end_turn`, matching this "
    "schema:\n"
    "{\n"
    '  "per_axis": {"<axis>": {"groundedness": "ok"|"thin"|"unsupported", "notes": str, "missed_evidence": [str]}},\n'
    '  "cross_axis": [str],          # observations spanning multiple axes\n'
    '  "flagged_claims": [{"source_axis": str, "claim": str, "flag": "unsupported"|"weakly_supported"|"contradicted", "reason": str}]\n'
    "}\n\n"
    "Do NOT fabricate concerns. If everything is supported, return empty arrays for flagged_claims and cross_axis."
)


def _format_axis_results_for_critic(results: list[AxisResult]) -> str:
    lines: list[str] = []
    for r in results:
        cite_summary = ", ".join(
            f"{c.kind}:{c.ticker}:{getattr(c, 'concept', getattr(c, 'chunk_id', '?'))}"
            for c in r.citations[:6]
        )
        lines.append(
            f"### {r.axis} (mode={r.mode}, grounding={r.grounding})\n"
            f"{r.comparison}\n"
            f"Citations: {cite_summary}"
        )
    return "\n\n".join(lines)


def _format_rebuttals_for_critic(rebuttals: list[Rebuttal]) -> str:
    lines = "\n".join(
        f"- [{r.source_axis}] {r.disposition}: {r.claim} ({r.reason})" for r in rebuttals
    )
    return (
        "PRIOR REBUTTALS (the analyst already responded to these flags):\n"
        f"{lines}\n\n"
        "Do NOT re-flag a claim that was retracted or adequately defended with a citation. "
        "Only flag NEW problems or defenses that remain unsupported."
    )


class _ToolDispatch:
    def __init__(
        self,
        *,
        pool: ConnectionPool,
        qdrant: QdrantClient,
        embed_query: Callable[[str], tuple[list[float], dict[str, float]]],
        trace_ctx: TraceCtx | None = None,
    ) -> None:
        self.pool = pool
        self.qdrant = qdrant
        self.embed_query = embed_query
        self.trace_ctx = trace_ctx

    def call(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        start = time.monotonic()
        ok, content = self._dispatch(name, args)
        if self.trace_ctx is not None:
            self.trace_ctx.event(
                f"tool:{name}",
                duration_ms=int((time.monotonic() - start) * 1000),
                error_kind="none" if ok else "transient",
                error_reason=None if ok else content[:240],
            )
        return ok, content

    def _dispatch(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        try:
            if name == "search_filings":
                dense, sparse = self.embed_query(str(args["query"]))
                hits = hybrid_search(
                    self.qdrant,
                    dense_vec=dense,
                    sparse_weights=sparse,
                    tickers=args.get("tickers"),
                    sections=args.get("sections"),
                    top_k=int(args.get("top_k", 5)),
                )
                payload = [
                    {
                        "chunk_id": h.chunk_id,
                        "score": h.score,
                        "ticker": h.payload.get("ticker"),
                        "section": h.payload.get("section"),
                        "text_excerpt": str(h.payload.get("text", ""))[:400],
                    }
                    for h in hits
                ]
                return True, json.dumps(payload)
            if name == "get_financial_concept":
                facts = get_financial_concept(
                    self.pool,
                    ticker=str(args["ticker"]),
                    key=str(args["key"]),
                    periods=args.get("periods"),
                )
                payload = [
                    {
                        "value": f.value,
                        "unit": f.unit,
                        "period": f.period,
                        "accession": f.accession,
                        "resolved_concept": f.resolved_concept,
                    }
                    for f in facts
                ]
                return True, json.dumps(payload)
            if name == "get_filing_section":
                section = get_filing_section(
                    self.qdrant,
                    ticker=str(args["ticker"]),
                    accession=str(args["accession"]),
                    section=str(args["section"]),
                )
                return True, json.dumps({"text": section.text[:6000]})
            return False, f"unknown tool: {name}"
        except Exception as e:  # noqa: BLE001
            return False, f"tool_error: {type(e).__name__}: {e}"


def _content_blocks(resp: Any) -> list[Any]:
    return list(getattr(resp, "content", []) or [])


def _stop_reason(resp: Any) -> str:
    return str(getattr(resp, "stop_reason", ""))


def _extract_final_json(resp: Any) -> dict[str, Any] | None:
    text = ""
    for b in _content_blocks(resp):
        if getattr(b, "type", "") == "text":
            text += str(getattr(b, "text", ""))
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
    s = s.strip()
    if not s:
        return None
    try:
        return dict(json.loads(s))
    except json.JSONDecodeError:
        return None


def critic(
    axis_results: list[AxisResult],
    *,
    sonnet_client: ChatClient,
    pool: ConnectionPool,
    qdrant: QdrantClient,
    embed_query: Callable[[str], tuple[list[float], dict[str, float]]],
    max_turns: int = CRITIC_MAX_TURNS,
    wall_clock_s: float = CRITIC_WALL_CLOCK_S,
    llm_cache: Cache | None = None,
    prompt_version: str = "critic-v1",
    trace_ctx: TraceCtx | None = None,
    rebuttals: list[Rebuttal] | None = None,
) -> Critique | None:
    # Phase 7.5 / decision 11: bounded agentic loop. Containment property -
    # any exception, hit-cap, or timeout returns None and the graph routes to
    # synthesize regardless.
    start = time.monotonic()
    deadline = start + wall_clock_s
    dispatch = _ToolDispatch(pool=pool, qdrant=qdrant, embed_query=embed_query, trace_ctx=trace_ctx)
    tool_calls: list[ToolCallRecord] = []

    content = (
        "AXIS RESULTS TO VERIFY:\n\n"
        + _format_axis_results_for_critic(axis_results)
        + "\n\nUse the tools to check claims, then emit the final JSON."
    )
    # Guarded so an empty rebuttal list leaves the prompt byte-identical to the
    # no-rebuttal arm (preserves its LLM cache and A/B baseline).
    if rebuttals:
        content += "\n\n" + _format_rebuttals_for_critic(rebuttals)
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

    final: dict[str, Any] | None = None
    turns = 0
    timed_out = False
    failed = False
    while turns < max_turns:
        if time.monotonic() >= deadline:
            timed_out = True
            break
        turns += 1
        try:
            resp = chat_maybe_cached(
                sonnet_client,
                llm_cache,
                prompt_version=prompt_version,
                system=[
                    {"type": "text", "text": _CRITIC_SYSTEM, "cache_control": {"type": "ephemeral"}}
                ],
                messages=messages,
                tools=_TOOL_DEFS,
                temperature=0.0,
                max_tokens=2048,
            )
        except Exception:  # noqa: BLE001
            failed = True
            break

        if trace_ctx is not None:
            trace_ctx.event(
                "llm:critic",
                **llm_trace_fields(sonnet_client.model, resp),
                input_shape={"turn": turns},
            )

        stop_reason = _stop_reason(resp)
        tool_uses = [b for b in _content_blocks(resp) if getattr(b, "type", "") == "tool_use"]

        # Append assistant turn to history so the next turn sees its prior reasoning.
        assistant_blocks: list[dict[str, Any]] = []
        for b in _content_blocks(resp):
            if getattr(b, "type", "") == "text":
                assistant_blocks.append({"type": "text", "text": str(getattr(b, "text", ""))})
            elif getattr(b, "type", "") == "tool_use":
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(getattr(b, "id", "")),
                        "name": str(getattr(b, "name", "")),
                        "input": dict(getattr(b, "input", {}) or {}),
                    }
                )
        messages.append({"role": "assistant", "content": assistant_blocks})

        if stop_reason == "end_turn" and not tool_uses:
            final = _extract_final_json(resp)
            break

        # Execute tools and append tool_result blocks for the next turn.
        if tool_uses:
            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                name = str(getattr(tu, "name", ""))
                args = dict(getattr(tu, "input", {}) or {})
                ok, content = dispatch.call(name, args)
                tool_calls.append(
                    ToolCallRecord(tool=name, args=args, ok=ok, result_summary=content[:400])
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(tu, "id", ""),
                        "content": content,
                        "is_error": not ok,
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            continue

        # No tool calls and no final JSON: ask once more politely.
        messages.append(
            {
                "role": "user",
                "content": "Emit the final JSON object now if you are done, or call a tool.",
            }
        )

    if failed:
        return None
    if final is None:
        # Containment fallback (hit cap, parse failure, or timeout).
        return Critique(
            status="timeout" if timed_out else ("failed" if not final else "partial"),
            per_axis={
                r.axis: AxisAssessment(axis=r.axis, groundedness="ok", notes="")
                for r in axis_results
            },
            cross_axis=[],
            flagged_claims=[],
            tool_calls=tool_calls,
            turns_used=turns,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    return Critique(
        status="ok",
        per_axis={
            axis: AxisAssessment(
                axis=axis,
                groundedness=str(body.get("groundedness", "ok")),  # type: ignore[arg-type]
                notes=str(body.get("notes", "")),
                missed_evidence=[str(x) for x in (body.get("missed_evidence") or [])],
            )
            for axis, body in (final.get("per_axis") or {}).items()
        },
        cross_axis=[str(x) for x in (final.get("cross_axis") or [])],
        flagged_claims=[
            FlaggedClaim(
                source_axis=str(fc.get("source_axis", "")),
                claim=str(fc.get("claim", "")),
                flag=str(fc.get("flag", "unsupported")),  # type: ignore[arg-type]
                reason=str(fc.get("reason", "")),
            )
            for fc in (final.get("flagged_claims") or [])
        ],
        tool_calls=tool_calls,
        turns_used=turns,
        duration_ms=int((time.monotonic() - start) * 1000),
    )
