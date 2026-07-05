from __future__ import annotations

from pathlib import Path

from quorum.cache.llm_cache import (
    build_llm_cache_key,
    cached_call,
    open_llm_cache,
)


def _msgs() -> list[dict[str, object]]:
    return [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_cache_hits_once_then_replays(tmp_cache_dir: Path) -> None:
    cache = open_llm_cache(tmp_cache_dir)
    key = build_llm_cache_key(
        model="claude-sonnet-4-6",
        prompt_version="v1",
        messages=_msgs(),
        temperature=0.0,
        max_tokens=128,
    )
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        return "ok"

    assert cached_call(cache, key, fn) == ("ok", False)
    assert cached_call(cache, key, fn) == ("ok", True)
    assert cached_call(cache, key, fn) == ("ok", True)
    assert len(calls) == 1, "underlying function should be invoked exactly once"


def test_prompt_version_invalidates(tmp_cache_dir: Path) -> None:
    k1 = build_llm_cache_key(
        model="m",
        prompt_version="v1",
        messages=_msgs(),
        temperature=0.0,
        max_tokens=128,
    )
    k2 = build_llm_cache_key(
        model="m",
        prompt_version="v2",
        messages=_msgs(),
        temperature=0.0,
        max_tokens=128,
    )
    assert k1 != k2


def test_key_stable_under_dict_reordering(tmp_cache_dir: Path) -> None:
    # Canonical-JSON encoder smoke gate at the cache-key level. Different
    # serializations of the same logical messages must produce the same key.
    a = [{"role": "user", "content": [{"type": "text", "text": "x", "annotations": []}]}]
    b = [{"role": "user", "content": [{"annotations": [], "text": "x", "type": "text"}]}]
    ka = build_llm_cache_key(
        model="m", prompt_version="v1", messages=a, temperature=0.0, max_tokens=64
    )
    kb = build_llm_cache_key(
        model="m", prompt_version="v1", messages=b, temperature=0.0, max_tokens=64
    )
    assert ka == kb


def test_different_models_collide_not(tmp_cache_dir: Path) -> None:
    k1 = build_llm_cache_key(
        model="claude-sonnet-4-6",
        prompt_version="v1",
        messages=_msgs(),
        temperature=0.0,
        max_tokens=128,
    )
    k2 = build_llm_cache_key(
        model="claude-haiku-4-5",
        prompt_version="v1",
        messages=_msgs(),
        temperature=0.0,
        max_tokens=128,
    )
    assert k1 != k2


def test_temperature_and_max_tokens_in_key(tmp_cache_dir: Path) -> None:
    base = dict(model="m", prompt_version="v1", messages=_msgs(), temperature=0.0, max_tokens=64)
    k_base = build_llm_cache_key(**base)  # type: ignore[arg-type]
    k_temp = build_llm_cache_key(**{**base, "temperature": 0.7})  # type: ignore[arg-type]
    k_mt = build_llm_cache_key(**{**base, "max_tokens": 128})  # type: ignore[arg-type]
    assert k_base != k_temp
    assert k_base != k_mt
