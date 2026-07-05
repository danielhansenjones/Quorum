from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Pricing:
    input: float  # USD per million input tokens
    output: float  # USD per million output tokens
    cache_write_5m: float  # USD per million 5-minute cache-write tokens
    cache_read: float  # USD per million cache-read tokens


# Anthropic public rates (claude.com/pricing), confirmed 2026-05-28. USD / MTok.
# Keyed by the exact model id the router reports. Local/unknown models are
# absent and cost $0 (no API spend).
PRICING: dict[str, Pricing] = {
    "claude-opus-4-8": Pricing(input=5.0, output=25.0, cache_write_5m=6.25, cache_read=0.50),
    "claude-sonnet-4-6": Pricing(input=3.0, output=15.0, cache_write_5m=3.75, cache_read=0.30),
    "claude-haiku-4-5": Pricing(input=1.0, output=5.0, cache_write_5m=1.25, cache_read=0.10),
}

_MILLION = 1_000_000.0


def extract_usage(resp: Any) -> dict[str, int]:
    # Returns the four token buckets, normalized across backends. Anthropic
    # exposes cache_creation/cache_read; the OpenAI/vLLM shape does not.
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    anthropic_input = getattr(u, "input_tokens", None)
    if anthropic_input is not None:
        return {
            "input": int(anthropic_input or 0),
            "output": int(getattr(u, "output_tokens", 0) or 0),
            "cache_write": int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            "cache_read": int(getattr(u, "cache_read_input_tokens", 0) or 0),
        }
    return {
        "input": int(getattr(u, "prompt_tokens", 0) or 0),
        "output": int(getattr(u, "completion_tokens", 0) or 0),
        "cache_write": 0,
        "cache_read": 0,
    }


def cost_dollars(model: str, usage: dict[str, int]) -> float:
    p = PRICING.get(model)
    if p is None:
        return 0.0
    return (
        usage["input"] * p.input
        + usage["cache_write"] * p.cache_write_5m
        + usage["cache_read"] * p.cache_read
        + usage["output"] * p.output
    ) / _MILLION


def llm_trace_fields(model: str, resp: Any, *, cache_hit: bool = False) -> dict[str, Any]:
    # Maps a chat response to TraceEvent token/cost fields. billed carries the
    # notional cost of the work even on a disk-cache replay so A/B pairing
    # stays comparable across warm and cold arms; effective is the real API
    # spend, which a replay does not incur.
    u = extract_usage(resp)
    c = cost_dollars(model, u)
    return {
        "tokens_in": u["input"] + u["cache_write"],
        "tokens_out": u["output"],
        "cache_read_tokens": u["cache_read"],
        "cost_dollars_billed": c,
        "cost_dollars_effective": 0.0 if cache_hit else c,
    }
