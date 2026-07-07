from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# Model IDs are locked by the design. Sonnet ID per the project brief; Haiku ID
# updates whenever the Anthropic SDK rotates the model alias.
SONNET_MODEL = "claude-sonnet-4-6"
HAIKU_MODEL = "claude-haiku-4-5"
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

Role = Literal[
    "classifier", "analyst", "synthesizer", "judge_dev", "judge_canonical", "judge_audit", "legwork"
]
Backend = Literal["anthropic", "vllm", "openai"]


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


@dataclass(slots=True)
class OpenAIChat:
    # Cross-family judge (self-preference measurement). Talks the same OpenAI wire
    # as VllmChat but against the real API with a keyed client. GPT-5 reasoning
    # models reject temperature/top_p and require max_completion_tokens, and
    # reasoning tokens count against that cap - so the visible-output budget is
    # padded or the JSON verdict truncates. usage tallies real (cache-miss) spend.
    model: str
    reasoning_effort: str = "none"
    reasoning_headroom: int = 3000
    backend: Backend = "openai"
    _create: Callable[..., Any] | None = None
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0}
    )

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._create = OpenAI().chat.completions.create

    def chat(self, **kwargs: Any) -> Any:
        assert self._create is not None
        kwargs.pop("temperature", None)
        visible = kwargs.pop("max_tokens", 700)
        kwargs["max_completion_tokens"] = visible + self.reasoning_headroom
        kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        resp = self._create(model=self.model, **kwargs)
        u = getattr(resp, "usage", None)
        if u is not None:
            self.usage["prompt"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion"] += getattr(u, "completion_tokens", 0) or 0
            details = getattr(u, "completion_tokens_details", None)
            self.usage["reasoning"] += getattr(details, "reasoning_tokens", 0) or 0
            self.usage["calls"] += 1
        choices = getattr(resp, "choices", None) or []
        if choices and getattr(choices[0], "finish_reason", None) == "length":
            # A truncated verdict parses to garbage and silently biases the audit.
            # Fail loud so the fix is more reasoning_headroom, not a bad number.
            raise RuntimeError(
                "audit judge hit the token cap (finish_reason=length); "
                "raise reasoning_headroom instead of accepting a truncated verdict"
            )
        return resp


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
    if role == "judge_audit":
        from quorum.config.settings import get_settings

        model = get_settings().audit_judge_model
        if not model:
            raise ValueError(
                "judge_audit requires AUDIT_JUDGE_MODEL (a pinned snapshot); unset means the "
                "cross-family audit is off and no OpenAI client is constructed"
            )
        return OpenAIChat(model=model)
    raise ValueError(f"unknown role: {role}")
