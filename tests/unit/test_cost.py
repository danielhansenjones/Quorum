from __future__ import annotations

from types import SimpleNamespace

from quorum.trace.cost import (
    PRICING,
    cost_dollars,
    extract_usage,
    llm_trace_fields,
)


def _anthropic_resp(inp: int, out: int, cw: int = 0, cr: int = 0) -> object:
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=inp,
            output_tokens=out,
            cache_creation_input_tokens=cw,
            cache_read_input_tokens=cr,
        )
    )


def _openai_resp(prompt: int, completion: int) -> object:
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    )


def test_extract_usage_anthropic() -> None:
    u = extract_usage(_anthropic_resp(100, 50, cw=20, cr=10))
    assert u == {"input": 100, "output": 50, "cache_write": 20, "cache_read": 10}


def test_extract_usage_openai() -> None:
    u = extract_usage(_openai_resp(200, 80))
    assert u == {"input": 200, "output": 80, "cache_write": 0, "cache_read": 0}


def test_extract_usage_missing() -> None:
    assert extract_usage(SimpleNamespace()) == {
        "input": 0,
        "output": 0,
        "cache_write": 0,
        "cache_read": 0,
    }


def test_cost_dollars_sonnet() -> None:
    # 1M input + 1M output at Sonnet 4.6 rates = $3 + $15.
    c = cost_dollars(
        "claude-sonnet-4-6",
        {"input": 1_000_000, "output": 1_000_000, "cache_write": 0, "cache_read": 0},
    )
    assert abs(c - 18.0) < 1e-9


def test_cost_dollars_includes_cache_buckets() -> None:
    # 1M cache-write at 3.75 + 1M cache-read at 0.30 = 4.05.
    c = cost_dollars(
        "claude-sonnet-4-6",
        {"input": 0, "output": 0, "cache_write": 1_000_000, "cache_read": 1_000_000},
    )
    assert abs(c - 4.05) < 1e-9


def test_cost_dollars_haiku() -> None:
    c = cost_dollars(
        "claude-haiku-4-5",
        {"input": 1_000_000, "output": 0, "cache_write": 0, "cache_read": 0},
    )
    assert abs(c - 1.0) < 1e-9


def test_cost_dollars_unknown_model_is_free() -> None:
    assert (
        cost_dollars(
            "qwen-2.5-7b-awq", {"input": 9_999, "output": 9_999, "cache_write": 0, "cache_read": 0}
        )
        == 0.0
    )


def test_llm_trace_fields_maps_buckets_and_cost() -> None:
    f = llm_trace_fields("claude-sonnet-4-6", _anthropic_resp(1_000_000, 0, cw=0, cr=0))
    assert f["tokens_in"] == 1_000_000
    assert f["tokens_out"] == 0
    assert f["cache_read_tokens"] == 0
    assert abs(f["cost_dollars_billed"] - 3.0) < 1e-9
    assert f["cost_dollars_billed"] == f["cost_dollars_effective"]


def test_llm_trace_fields_cache_hit_zeroes_effective_keeps_billed() -> None:
    # A disk-cache replay keeps the notional cost (A/B pairing across warm and
    # cold arms) but spent no API dollars.
    f = llm_trace_fields(
        "claude-sonnet-4-6", _anthropic_resp(1_000_000, 0, cw=0, cr=0), cache_hit=True
    )
    assert abs(f["cost_dollars_billed"] - 3.0) < 1e-9
    assert f["cost_dollars_effective"] == 0.0


def test_pricing_has_session_models() -> None:
    assert "claude-sonnet-4-6" in PRICING
    assert "claude-haiku-4-5" in PRICING
