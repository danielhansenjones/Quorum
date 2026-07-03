from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from diskcache import Cache

from quorum.cache.canonical import canonical_json, sha256_hex


def build_llm_cache_key(
    *,
    model: str,
    prompt_version: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    extras: dict[str, Any] | None = None,
) -> str:
    # The architecture's contract: sha256 over canonical-JSON of the full
    # multi-turn message list. The critic node's later turns include tool-result
    # payloads, so message-level (not single-prompt) hashing is required.
    # extras carries everything else that shapes the response (system prompt,
    # tool schemas), so editing a prompt or tool without bumping prompt_version
    # misses instead of silently replaying responses produced under the old one.
    messages_hash = sha256_hex(canonical_json(messages))
    key_obj = {
        "model": model,
        "prompt_version": prompt_version,
        "messages_hash": messages_hash,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extras_hash": sha256_hex(canonical_json(extras)) if extras else None,
    }
    return sha256_hex(canonical_json(key_obj))


def open_llm_cache(cache_dir: Path) -> Cache:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Cache(str(cache_dir))


def cached_call[T](cache: Cache, key: str, fn: Callable[[], T]) -> T:
    hit = cache.get(key, default=_MISS)
    if hit is _MISS:
        result = fn()
        cache[key] = result
        return result
    return hit  # type: ignore[no-any-return]


class _MissSentinel:
    __slots__ = ()


_MISS = _MissSentinel()
