from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from diskcache import Cache

from quorum.graph.nodes.analyze_axis import _write_axis_result
from quorum.state.axis import AxisTask


# Module-level so diskcache can pickle the cached response.
class _Block:
    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


_GOOD = (
    '{"comparison": "AAPL leads", "per_company": {"AAPL": "solid"}, '
    '"grounding": "ok", "reason_if_not_ok": ""}'
)


class _MalformedThenGood:
    backend = "anthropic"
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return _Resp("Sure! Here is prose without any JSON object at all.")
        return _Resp(_GOOD)


def _task() -> AxisTask:
    return AxisTask(axis="profitability", mode="semantic", tickers=["AAPL"], query_or_concept="q")


def test_parse_failure_retry_feeds_error_back(tmp_path: Path) -> None:
    # The retry must not be byte-identical: at temperature 0 (and through the
    # disk cache) an identical request replays the same malformed response.
    client = _MalformedThenGood()
    cache = Cache(str(tmp_path / "c"))
    res = _write_axis_result(
        _task(),
        quant_evidence={},
        qual_evidence={"AAPL": []},
        sonnet_client=client,  # type: ignore[arg-type]
        deadline=time.monotonic() + 30,
        llm_cache=cache,
    )
    assert len(client.calls) == 2, "second attempt must miss the cache and reach the model"
    retry_messages = client.calls[1]["messages"]
    assert [m["role"] for m in retry_messages] == ["user", "assistant", "user"]
    assert "could not be parsed" in retry_messages[2]["content"]
    assert res.grounding == "ok"
    assert res.attempts == 2


def test_transport_error_retries_identical_request(tmp_path: Path) -> None:
    class FailsOnceThenGood:
        backend = "anthropic"
        model = "claude-sonnet-4-6"

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def chat(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("anthropic 529")
            return _Resp(_GOOD)

    client = FailsOnceThenGood()
    res = _write_axis_result(
        _task(),
        quant_evidence={},
        qual_evidence={"AAPL": []},
        sonnet_client=client,  # type: ignore[arg-type]
        deadline=time.monotonic() + 30,
        llm_cache=None,
    )
    assert len(client.calls) == 2
    assert client.calls[0]["messages"] == client.calls[1]["messages"]
    assert res.grounding == "ok"
