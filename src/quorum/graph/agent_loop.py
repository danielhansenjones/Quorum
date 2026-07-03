from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from diskcache import Cache

from quorum.models.cached_chat import chat_maybe_cached
from quorum.models.router import ChatClient
from quorum.state.critique import ToolCallRecord
from quorum.trace.cost import llm_trace_fields
from quorum.trace.writer import TraceCtx

# Phase 13c. A general bounded tool-use loop: a model picks tools per turn, a
# caller-supplied dispatch executes them, and the loop stops on end_turn, a turn
# cap, a wall-clock deadline, or a model error. The legwork phase of the agentic
# analyst runs on this. (The critic has its own equivalent loop; unifying the
# two onto this helper is a deliberate post-campaign-freeze follow-up so it
# cannot perturb the +critic A/B arm.)

Dispatch = Callable[[str, dict[str, Any]], tuple[bool, str]]


@dataclass(slots=True)
class AgentLoopResult:
    final_text: str
    tool_calls: list[ToolCallRecord]
    turns_used: int
    stop: str  # "end_turn" | "cap" | "timeout" | "failed"
    error: str | None = None


def _blocks(resp: Any) -> list[Any]:
    return list(getattr(resp, "content", []) or [])


def _stop_reason(resp: Any) -> str:
    return str(getattr(resp, "stop_reason", ""))


def run_agent_loop(
    *,
    client: ChatClient,
    system: Any,
    tools: list[dict[str, Any]],
    dispatch: Dispatch,
    initial_user: str,
    max_turns: int,
    wall_clock_s: float,
    label: str = "agent",
    max_tokens: int = 2048,
    llm_cache: Cache | None = None,
    prompt_version: str = "agent-v1",
    trace_ctx: TraceCtx | None = None,
) -> AgentLoopResult:
    if client.backend != "anthropic":
        # The loop builds Anthropic wire-format requests (top-level `system`,
        # `input_schema` tools, tool_result blocks); an OpenAI-protocol client
        # would raise on turn 1. Fail fast with a reason the caller can surface
        # instead of a swallowed TypeError.
        return AgentLoopResult(
            final_text="",
            tool_calls=[],
            turns_used=0,
            stop="failed",
            error=f"agent loop requires an anthropic-protocol client, got {client.backend!r}",
        )

    deadline = time.monotonic() + wall_clock_s
    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_user}]
    tool_calls: list[ToolCallRecord] = []
    final_text = ""
    turns = 0
    stop = "cap"
    error: str | None = None
    while turns < max_turns:
        if time.monotonic() >= deadline:
            stop = "timeout"
            break
        turns += 1
        try:
            resp = chat_maybe_cached(
                client,
                llm_cache,
                prompt_version=prompt_version,
                system=system,
                messages=messages,
                tools=tools,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            stop = "failed"
            error = f"{type(e).__name__}: {e}"
            break
        if trace_ctx is not None:
            trace_ctx.event(
                f"llm:{label}", **llm_trace_fields(client.model, resp), input_shape={"turn": turns}
            )
        blocks = _blocks(resp)
        tool_uses = [b for b in blocks if getattr(b, "type", "") == "tool_use"]
        assistant_blocks: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for b in blocks:
            kind = getattr(b, "type", "")
            if kind == "text":
                txt = str(getattr(b, "text", ""))
                text_parts.append(txt)
                assistant_blocks.append({"type": "text", "text": txt})
            elif kind == "tool_use":
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(getattr(b, "id", "")),
                        "name": str(getattr(b, "name", "")),
                        "input": dict(getattr(b, "input", {}) or {}),
                    }
                )
        messages.append({"role": "assistant", "content": assistant_blocks})

        if _stop_reason(resp) == "end_turn" and not tool_uses:
            final_text = "".join(text_parts)
            stop = "end_turn"
            break

        if tool_uses:
            results: list[dict[str, Any]] = []
            for tu in tool_uses:
                name = str(getattr(tu, "name", ""))
                args = dict(getattr(tu, "input", {}) or {})
                ok, content = dispatch(name, args)
                tool_calls.append(
                    ToolCallRecord(tool=name, args=args, ok=ok, result_summary=content[:400])
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(tu, "id", ""),
                        "content": content,
                        "is_error": not ok,
                    }
                )
            messages.append({"role": "user", "content": results})
            continue

        messages.append({"role": "user", "content": "Call a tool or stop with your conclusion."})

    return AgentLoopResult(
        final_text=final_text, tool_calls=tool_calls, turns_used=turns, stop=stop, error=error
    )
