from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

# Fake responses are pickled verbatim into diskcache; module-level classes keep
# them picklable across the runner subprocesses that write and read the cache.


class KRUsage:
    def __init__(self) -> None:
        self.input_tokens = 120
        self.output_tokens = 80
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class KRBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class KRResponse:
    def __init__(self, text: str) -> None:
        self.content = [KRBlock(text)]
        self.stop_reason = "end_turn"
        self.usage = KRUsage()


class KRToolUseBlock:
    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class KRToolUseResponse:
    def __init__(self, block: KRToolUseBlock) -> None:
        self.content = [block]
        self.stop_reason = "tool_use"
        self.usage = KRUsage()


CLASSIFY_JSON = json.dumps(
    {
        "companies_raw": ["Apple", "Microsoft"],
        "axes": ["profitability", "growth", "risk_factors"],
        "out_of_scope": False,
        "reason": "",
    }
)

CRITIC_JSON = json.dumps(
    {
        "per_axis": {
            "profitability": {"groundedness": "ok", "notes": ""},
            "growth": {"groundedness": "ok", "notes": ""},
            "risk_factors": {"groundedness": "ok", "notes": ""},
        },
        "cross_axis": [],
        "flagged_claims": [],
    }
)

# No digits outside [TICKER:ID] markers: the synthesizer strips any line that
# carries a digit or "$" without a citation marker, and a stripped line would
# break the byte-equality assertions.
SYNTH_TEXT = (
    "## Comparison\n\n"
    "AAPL leads on profitability and growth. [AAPL:Q0]\n"
    "MSFT carries lower leverage risk. [MSFT:S0]"
)


def analyst_json(axis: str, grounding: str = "ok") -> str:
    return json.dumps(
        {
            "comparison": f"AAPL and MSFT track each other on {axis}. [AAPL:Q0] [MSFT:Q0]",
            "per_company": {
                "AAPL": f"AAPL holds steady on {axis}. [AAPL:Q0]",
                "MSFT": f"MSFT holds steady on {axis}. [MSFT:Q0]",
            },
            "grounding": grounding,
            "reason_if_not_ok": "" if grounding == "ok" else "evidence thin on one side",
        }
    )


def _analyst_grounding(axis: str, messages: list[dict[str, Any]]) -> str:
    # T4 script: the flagged axis grades weak on its first prompt variant, ok on
    # the revised one. revise_plan flips a weak structured axis to semantic, so
    # the prompt's "mode:" line identifies the variant by content; keying on
    # call order would desync under disk-cache replays.
    if os.environ.get("KR_WEAK_FIRST_AXIS") != axis:
        return "ok"
    content = messages[0].get("content", "")
    return "weak" if "\nmode: structured\n" in content else "ok"


def _tool_result_rounds(messages: list[dict[str, Any]]) -> int:
    rounds = 0
    for m in messages:
        content = m.get("content")
        if m.get("role") != "user" or not isinstance(content, list):
            continue
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            rounds += 1
    return rounds


def _critic_response(messages: list[dict[str, Any]]) -> KRResponse | KRToolUseResponse:
    if os.environ.get("KR_CRITIC_TOOLS") != "1":
        return KRResponse(CRITIC_JSON)
    # T11 script: a 3-turn tool loop keyed on how many tool_result exchanges the
    # history already carries, never on call order. Tool-use ids are constants,
    # not uuids: the id feeds the next turn's message history, which feeds the
    # cache key, so an unstable id would miss the cache across runs.
    rounds = _tool_result_rounds(messages)
    if rounds == 0:
        return KRToolUseResponse(
            KRToolUseBlock(
                "kr-tool-1",
                "get_financial_concept",
                {"ticker": "AAPL", "key": "profitability.revenue"},
            )
        )
    if rounds == 1:
        return KRToolUseResponse(
            KRToolUseBlock(
                "kr-tool-2",
                "search_filings",
                {"query": "risk factors", "tickers": ["AAPL"]},
            )
        )
    return KRResponse(CRITIC_JSON)


def route_of(messages: list[dict[str, Any]]) -> str:
    # Route on the first user message, not call order: disk-cache replays skip
    # fake chat() calls entirely, so any call-count script would desync.
    content = messages[0].get("content", "")
    if isinstance(content, str):
        if content.startswith("axis: "):
            axis = content.splitlines()[0].removeprefix("axis: ").strip()
            return f"analyst:{axis}"
        if content.startswith("AXIS RESULTS TO VERIFY:"):
            return "critic"
        if content.startswith("Question:"):
            return "synthesize"
    return "unknown"


_HOOK_LOCK = threading.Lock()
_HOOK_COUNTS: dict[str, int] = {}


def parse_hook_spec(spec: str) -> tuple[str, int]:
    point, sep, n = spec.rpartition("#")
    if sep and n.isdigit():
        return point, int(n)
    return spec, 1


def _maybe_hook(point: str) -> None:
    spec = os.environ.get("KR_HOOK")
    if not spec:
        return
    want_point, want_n = parse_hook_spec(spec)
    if point != want_point:
        return
    with _HOOK_LOCK:
        _HOOK_COUNTS[point] = _HOOK_COUNTS.get(point, 0) + 1
        arrival = _HOOK_COUNTS[point]
    if arrival != want_n:
        return
    (Path(os.environ["KR_HOOK_DIR"]) / "hook.hit").write_text(point)
    # Block forever; the parent test observes hook.hit and SIGKILLs us.
    while True:
        time.sleep(60)


def _log_call(route: str) -> None:
    path = os.environ.get("KR_CALLS_LOG")
    if not path:
        return
    with open(path, "a") as f:
        f.write(route + "\n")


class FakeClassifier:
    backend = "anthropic"
    model = "fake-classifier"

    def chat(self, **kwargs: Any) -> KRResponse:
        _log_call("classify")
        _maybe_hook("in_chat:classify")
        return KRResponse(CLASSIFY_JSON)


class FakeSonnet:
    backend = "anthropic"
    model = "claude-sonnet-4-6"

    def chat(self, **kwargs: Any) -> KRResponse | KRToolUseResponse:
        route = route_of(kwargs["messages"])
        _log_call(route)
        _maybe_hook(f"in_chat:{route}")
        if route.startswith("analyst:"):
            axis = route.split(":", 1)[1]
            return KRResponse(analyst_json(axis, _analyst_grounding(axis, kwargs["messages"])))
        if route == "critic":
            return _critic_response(kwargs["messages"])
        if route == "synthesize":
            return KRResponse(SYNTH_TEXT)
        raise ValueError(f"unroutable sonnet call: {kwargs['messages'][0]!r}")


class KRPoint:
    def __init__(self, id: str, score: float, payload: dict[str, Any]) -> None:
        self.id = id
        self.score = score
        self.payload = payload


class KRPointsResult:
    def __init__(self, points: list[KRPoint]) -> None:
        self.points = points


class FakeQdrant:
    def query_points(self, **kwargs: Any) -> KRPointsResult:
        # Two canned hits regardless of filters; text carries no digits so the
        # synthesizer's uncited-number strip never fires on quoted passages.
        return KRPointsResult(
            [
                KRPoint(
                    "kr-chunk-a",
                    0.9,
                    {
                        "text": "Competition remains intense across hardware and services.",
                        "chunk_id": "kr-chunk-a",
                        "section": "item_1a_risk_factors",
                        "accession": "0000-kr-1",
                        "ticker": "AAPL",
                    },
                ),
                KRPoint(
                    "kr-chunk-b",
                    0.8,
                    {
                        "text": "Cloud demand may fluctuate with enterprise spending cycles.",
                        "chunk_id": "kr-chunk-b",
                        "section": "item_1a_risk_factors",
                        "accession": "0000-kr-1",
                        "ticker": "MSFT",
                    },
                ),
            ]
        )


def embed_query_stub(text: str) -> tuple[list[float], dict[str, float]]:
    return [0.0] * 1024, {}
