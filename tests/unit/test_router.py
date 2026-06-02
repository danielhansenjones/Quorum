from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from quorum.cache.llm_cache import open_llm_cache
from quorum.models.cached_chat import cached_chat, fake_client
from quorum.models.router import (
    DEFAULT_VLLM_MODEL,
    HAIKU_MODEL,
    SONNET_MODEL,
    AnthropicChat,
    VllmChat,
    get_client,
)


def test_classifier_vllm_when_url_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid actually constructing the OpenAI client (it imports openai which is
    # fine, but we do not want a network probe). Monkeypatch the post-init.
    monkeypatch.setattr(VllmChat, "__post_init__", lambda self: None)
    c = get_client("classifier", vllm_url="http://vllm:8000")
    assert c.backend == "vllm"
    assert c.model == DEFAULT_VLLM_MODEL


def test_classifier_haiku_when_vllm_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AnthropicChat, "__post_init__", lambda self: None)
    c = get_client("classifier", vllm_url=None)
    assert c.backend == "anthropic"
    assert c.model == HAIKU_MODEL


def test_judge_dev_follows_classifier_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VllmChat, "__post_init__", lambda self: None)
    c = get_client("judge_dev", vllm_url="http://vllm:8000")
    assert c.backend == "vllm"


def test_analyst_always_sonnet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AnthropicChat, "__post_init__", lambda self: None)
    # Even with vllm_url set, analyst must be Sonnet (no fallback path).
    c1 = get_client("analyst", vllm_url=None)
    c2 = get_client("analyst", vllm_url="http://vllm:8000")
    c3 = get_client("synthesizer", vllm_url="http://vllm:8000")
    c4 = get_client("judge_canonical", vllm_url=None)
    assert c1.model == SONNET_MODEL
    assert c2.model == SONNET_MODEL
    assert c3.model == SONNET_MODEL
    assert c4.model == SONNET_MODEL


def test_cache_hits_across_roles_same_model(tmp_cache_dir: Path) -> None:
    # The contract: two role-resolved clients pointing at the same model share
    # cache keys. Here both "analyst" and "synthesizer" route to Sonnet.
    cache = open_llm_cache(tmp_cache_dir)
    call_count = {"n": 0}

    def make_response() -> dict[str, Any]:
        call_count["n"] += 1
        return {"text": "ok"}

    c1 = fake_client(SONNET_MODEL, make_response)
    c2 = fake_client(SONNET_MODEL, make_response)

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    r1 = cached_chat(c1, cache, prompt_version="v1", messages=msgs, temperature=0.0, max_tokens=64)
    r2 = cached_chat(c2, cache, prompt_version="v1", messages=msgs, temperature=0.0, max_tokens=64)
    assert r1 == r2
    assert call_count["n"] == 1, "second client with same model+args should hit cache"


def test_invalid_role_raises() -> None:
    with pytest.raises(ValueError):
        get_client("bogus")  # type: ignore[arg-type]


def test_legwork_vllm_when_url_set() -> None:
    c = get_client("legwork", vllm_url="http://vllm:8000")
    assert c.backend == "vllm"


def test_legwork_haiku_when_vllm_unset() -> None:
    c = get_client("legwork", vllm_url=None)
    assert c.backend == "anthropic"
    assert c.model == HAIKU_MODEL
