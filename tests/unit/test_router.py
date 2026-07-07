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
    OpenAIChat,
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
    r1, _ = cached_chat(
        c1, cache, prompt_version="v1", messages=msgs, temperature=0.0, max_tokens=64
    )
    r2, _ = cached_chat(
        c2, cache, prompt_version="v1", messages=msgs, temperature=0.0, max_tokens=64
    )
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


def test_openai_chat_reasoning_param_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    # gpt-5 reasoning models 400 on temperature and require max_completion_tokens;
    # reasoning tokens count against that cap, so the caller's visible-output
    # budget is padded rather than passed through. usage tallies real spend.
    monkeypatch.setattr(OpenAIChat, "__post_init__", lambda self: None)
    captured: dict[str, Any] = {}

    def fake(**kwargs: Any) -> Any:
        captured.update(kwargs)
        details = type("D", (), {"reasoning_tokens": 600})()
        usage = type(
            "U",
            (),
            {"prompt_tokens": 1200, "completion_tokens": 800, "completion_tokens_details": details},
        )()
        return type("R", (), {"usage": usage})()

    c = OpenAIChat(model="gpt-5.1")
    c._create = fake
    c.chat(messages=[{"role": "user", "content": "hi"}], max_tokens=700, temperature=0.0)

    assert c.backend == "openai"  # judges route non-anthropic backends through the OpenAI wire path
    assert "temperature" not in captured
    assert "max_tokens" not in captured
    assert captured["max_completion_tokens"] == 3700  # 700 visible + 3000 reasoning headroom
    assert (
        captured["reasoning_effort"] == "none"
    )  # audit default: comparable to non-reasoning Sonnet
    assert captured["model"] == "gpt-5.1"
    assert c.usage == {"prompt": 1200, "completion": 800, "reasoning": 600, "calls": 1}


def test_judge_audit_unset_constructs_no_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unset audit model: the role is unavailable, no OpenAI client is built, no
    # network is possible. A sentinel proves construction never happens.
    import quorum.config.settings as settings_mod

    monkeypatch.setattr(
        settings_mod, "get_settings", lambda: type("S", (), {"audit_judge_model": None})()
    )

    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("OpenAIChat must not be constructed when the audit is unset")

    monkeypatch.setattr("quorum.models.router.OpenAIChat", _boom)
    with pytest.raises(ValueError, match="AUDIT_JUDGE_MODEL"):
        get_client("judge_audit")

    monkeypatch.setattr(AnthropicChat, "__post_init__", lambda self: None)
    assert get_client("judge_dev", vllm_url=None).backend == "anthropic"


def test_judge_audit_uses_pinned_snapshot_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import quorum.config.settings as settings_mod

    monkeypatch.setattr(
        settings_mod,
        "get_settings",
        lambda: type("S", (), {"audit_judge_model": "gpt-5.1-2025-01-01"})(),
    )
    monkeypatch.setattr(OpenAIChat, "__post_init__", lambda self: None)
    c = get_client("judge_audit")
    assert c.backend == "openai"
    assert c.model == "gpt-5.1-2025-01-01"
    assert c.reasoning_effort == "none"


def test_openai_chat_fails_loud_on_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    # A length-truncated verdict parses to garbage; the client must raise, not
    # return it, so the audit never records a silently biased score.
    monkeypatch.setattr(OpenAIChat, "__post_init__", lambda self: None)

    def fake(**kwargs: Any) -> Any:
        choice = type("C", (), {"finish_reason": "length"})()
        return type("R", (), {"usage": None, "choices": [choice]})()

    c = OpenAIChat(model="gpt-5.1")
    c._create = fake
    with pytest.raises(RuntimeError, match="finish_reason=length"):
        c.chat(messages=[{"role": "user", "content": "hi"}], max_tokens=700)


def test_reasoning_effort_splits_cache_key(tmp_cache_dir: Path) -> None:
    # Two audit runs at different efforts must not collide on one cache key and
    # serve each other's verdicts.
    cache = open_llm_cache(tmp_cache_dir)
    calls = {"n": 0}

    class _Client:
        backend = "openai"

        def __init__(self, effort: str) -> None:
            self.model = "gpt-5.1"
            self.reasoning_effort = effort

        def chat(self, **k: Any) -> Any:
            calls["n"] += 1
            return {"text": "ok"}

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    cached_chat(
        _Client("none"), cache, prompt_version="v1", messages=msgs, temperature=0.0, max_tokens=64
    )
    cached_chat(
        _Client("high"), cache, prompt_version="v1", messages=msgs, temperature=0.0, max_tokens=64
    )
    assert calls["n"] == 2, "different reasoning_effort must produce a different cache key"
