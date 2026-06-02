from __future__ import annotations

from pathlib import Path
from typing import Any

from diskcache import Cache

from quorum.models.cached_chat import chat_maybe_cached, fake_client


class _Counter:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> dict[str, Any]:
        self.n += 1
        return {"call": self.n}


def _args() -> dict[str, Any]:
    return {
        "prompt_version": "analyst-v1",
        "system": [{"type": "text", "text": "sys"}],
        "messages": [{"role": "user", "content": "same question"}],
        "temperature": 0.0,
        "max_tokens": 100,
    }


def test_cache_hit_on_second_identical_call(tmp_path: Path) -> None:
    counter = _Counter()
    client = fake_client("claude-sonnet-4-6", counter)
    cache = Cache(str(tmp_path / "c"))

    first = chat_maybe_cached(client, cache, **_args())
    second = chat_maybe_cached(client, cache, **_args())

    assert counter.n == 1  # second served from cache
    assert first == second


def test_distinct_messages_miss(tmp_path: Path) -> None:
    counter = _Counter()
    client = fake_client("claude-sonnet-4-6", counter)
    cache = Cache(str(tmp_path / "c"))

    a = _args()
    chat_maybe_cached(client, cache, **a)
    b = dict(a)
    b["messages"] = [{"role": "user", "content": "different"}]
    chat_maybe_cached(client, cache, **b)

    assert counter.n == 2


def test_prompt_version_separates_keys(tmp_path: Path) -> None:
    counter = _Counter()
    client = fake_client("claude-sonnet-4-6", counter)
    cache = Cache(str(tmp_path / "c"))

    a = _args()
    chat_maybe_cached(client, cache, **a)
    b = dict(a)
    b["prompt_version"] = "analyst-v2"
    chat_maybe_cached(client, cache, **b)

    assert counter.n == 2


def test_no_cache_calls_every_time() -> None:
    counter = _Counter()
    client = fake_client("claude-sonnet-4-6", counter)

    chat_maybe_cached(client, None, **_args())
    chat_maybe_cached(client, None, **_args())

    assert counter.n == 2
