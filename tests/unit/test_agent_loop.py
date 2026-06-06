from __future__ import annotations

import time
from typing import Any

from quorum.graph.agent_loop import run_agent_loop


class _Block:
    def __init__(self, type: str, **kw: Any) -> None:
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, content: list[Any], stop_reason: str) -> None:
        self.content = content
        self.stop_reason = stop_reason


class _Scripted:
    backend = "anthropic"
    model = "claude-haiku-4-5"

    def __init__(self, responses: list[_Resp]) -> None:
        self._responses = responses
        self.calls = 0

    def chat(self, **kwargs: Any) -> Any:
        r = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return r


_TOOLS = [{"name": "search_filings", "description": "x", "input_schema": {"type": "object"}}]


def _ok(name: str, args: dict[str, Any]) -> tuple[bool, str]:
    return True, "result"


def test_agent_loop_end_turn_first_turn() -> None:
    client = _Scripted([_Resp([_Block("text", text="done summary")], "end_turn")])
    out = run_agent_loop(
        client=client,
        system="s",
        tools=_TOOLS,
        dispatch=_ok,
        initial_user="go",
        max_turns=5,
        wall_clock_s=10.0,
    )
    assert out.stop == "end_turn"
    assert out.final_text == "done summary"
    assert out.turns_used == 1
    assert out.tool_calls == []


def test_agent_loop_tool_then_end() -> None:
    client = _Scripted(
        [
            _Resp(
                [_Block("tool_use", id="t1", name="search_filings", input={"query": "q"})],
                "tool_use",
            ),
            _Resp([_Block("text", text="ok")], "end_turn"),
        ]
    )
    calls: list[str] = []

    def _record(name: str, args: dict[str, Any]) -> tuple[bool, str]:
        calls.append(name)
        return True, "r"

    out = run_agent_loop(
        client=client,
        system="s",
        tools=_TOOLS,
        dispatch=_record,
        initial_user="go",
        max_turns=5,
        wall_clock_s=10.0,
    )
    assert out.stop == "end_turn"
    assert out.turns_used == 2
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].tool == "search_filings"
    assert calls == ["search_filings"]


def test_agent_loop_hits_turn_cap() -> None:
    client = _Scripted(
        [
            _Resp(
                [_Block("tool_use", id="t", name="search_filings", input={"query": "q"})],
                "tool_use",
            )
        ]
    )
    out = run_agent_loop(
        client=client,
        system="s",
        tools=_TOOLS,
        dispatch=_ok,
        initial_user="go",
        max_turns=3,
        wall_clock_s=10.0,
    )
    assert out.stop == "cap"
    assert out.turns_used == 3


def test_agent_loop_wall_clock_timeout() -> None:
    class _Slow:
        backend = "anthropic"
        model = "claude-haiku-4-5"

        def chat(self, **kwargs: Any) -> Any:
            time.sleep(0.6)
            return _Resp(
                [_Block("tool_use", id="t", name="search_filings", input={"query": "q"})],
                "tool_use",
            )

    out = run_agent_loop(
        client=_Slow(),
        system="s",
        tools=_TOOLS,
        dispatch=_ok,
        initial_user="go",
        max_turns=5,
        wall_clock_s=0.5,
    )
    assert out.stop == "timeout"


def test_agent_loop_model_error() -> None:
    class _Raise:
        backend = "anthropic"
        model = "claude-haiku-4-5"

        def chat(self, **kwargs: Any) -> Any:
            raise RuntimeError("503")

    out = run_agent_loop(
        client=_Raise(),
        system="s",
        tools=_TOOLS,
        dispatch=_ok,
        initial_user="go",
        max_turns=5,
        wall_clock_s=10.0,
    )
    assert out.stop == "failed"
    assert out.turns_used == 1
