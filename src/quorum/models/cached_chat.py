from __future__ import annotations

from collections.abc import Callable
from typing import Any

from diskcache import Cache

from quorum.cache.llm_cache import build_llm_cache_key, cached_call
from quorum.models.router import ChatClient


def cached_chat(
    client: ChatClient,
    cache: Cache,
    *,
    prompt_version: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    chat_kwargs: dict[str, Any] | None = None,
) -> Any:
    # Cache key uses (client.model, prompt_version, messages, temperature, max_tokens).
    # Two different roles that resolve to the same model share keys; that is the
    # Phase 2c "cache hit across role types using the same model" contract.
    key = build_llm_cache_key(
        model=client.model,
        prompt_version=prompt_version,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    def call() -> Any:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if chat_kwargs:
            kwargs.update(chat_kwargs)
        return client.chat(**kwargs)

    return cached_call(cache, key, call)


def chat_maybe_cached(
    client: ChatClient,
    cache: Cache | None,
    *,
    prompt_version: str,
    system: Any,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: Any | None = None,
) -> Any:
    # One branch point for every node's chat call: passthrough when no cache is
    # configured, cached otherwise. `system` (and `tools`) ride in chat_kwargs so
    # they reach client.chat but stay out of the cache key - the key is
    # (model, prompt_version, messages, temperature, max_tokens), and
    # prompt_version is what distinguishes a changed system prompt or tool set.
    extra: dict[str, Any] = {"system": system}
    if tools is not None:
        extra["tools"] = tools
    if cache is None:
        return client.chat(
            messages=messages, temperature=temperature, max_tokens=max_tokens, **extra
        )
    return cached_chat(
        client,
        cache,
        prompt_version=prompt_version,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        chat_kwargs=extra,
    )


def fake_client(model: str, response_factory: Callable[[], Any]) -> ChatClient:
    # Test seam: the router clients always reach for network on first construction.
    # For unit tests we want to inject a pure function. This factory builds a
    # minimal client that satisfies the ChatClient protocol.
    class _Fake:
        backend = "anthropic"
        model = ""

        def __init__(self, m: str) -> None:
            self.model = m

        def chat(self, **kwargs: Any) -> Any:
            return response_factory()

    return _Fake(model)  # type: ignore[return-value]
