from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

# Model IDs are locked by the design. Sonnet ID per the project brief; Haiku ID
# updates whenever the Anthropic SDK rotates the model alias.
SONNET_MODEL = "claude-sonnet-4-6"
HAIKU_MODEL = "claude-haiku-4-5"
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

Role = Literal["classifier", "analyst", "synthesizer", "judge_dev", "judge_canonical", "legwork"]
Backend = Literal["anthropic", "vllm"]


class ChatClient(Protocol):
    backend: Backend
    model: str

    def chat(self, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class AnthropicChat:
    model: str
    backend: Backend = "anthropic"
    _create: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        # Lazy import so unit tests that never instantiate this client do not
        # require network credentials. The actual client is constructed once and
        # cached on the instance.
        from anthropic import Anthropic

        self._create = Anthropic().messages.create

    def chat(self, **kwargs: Any) -> Any:
        assert self._create is not None
        return self._create(model=self.model, **kwargs)


@dataclass(slots=True)
class VllmChat:
    base_url: str
    model: str
    backend: Backend = "vllm"
    _create: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._create = OpenAI(base_url=self.base_url, api_key="not-used").chat.completions.create

    def chat(self, **kwargs: Any) -> Any:
        assert self._create is not None
        return self._create(model=self.model, **kwargs)


def get_client(
    role: Role,
    *,
    vllm_url: str | None = None,
    vllm_model: str = DEFAULT_VLLM_MODEL,
) -> ChatClient:
    # Routing contract (Phase 2c gates):
    #   classifier, judge_dev, legwork -> vLLM if VLLM_URL set, Haiku fallback.
    #   analyst, synthesizer, judge_canonical -> Sonnet, no fallback.
    # legwork (Phase 13c) is the cheap tool-loop tier: it orchestrates retrieval
    # for the agentic analyst; the faithfulness-critical write stays on Sonnet.
    if role in ("classifier", "judge_dev", "legwork"):
        if vllm_url:
            return VllmChat(base_url=vllm_url, model=vllm_model)
        return AnthropicChat(model=HAIKU_MODEL)
    if role in ("analyst", "synthesizer", "judge_canonical"):
        return AnthropicChat(model=SONNET_MODEL)
    raise ValueError(f"unknown role: {role}")
