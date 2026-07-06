from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from quorum.graph.axis_config import SUPPORTED_AXES, AxisName
from quorum.models.router import ChatClient
from quorum.trace.cost import llm_trace_fields
from quorum.trace.writer import TraceCtx


class ClassifyOutput(BaseModel):
    companies_raw: list[str] = Field(default_factory=list)
    axes: list[AxisName] = Field(default_factory=list)
    out_of_scope: bool = False
    reason: str = ""


_SYSTEM = (
    "You classify financial research questions for the Quorum agent. Extract:\n"
    "  - companies_raw: company mentions exactly as written in the question.\n"
    "  - axes: comparison dimensions from this fixed set: " + ", ".join(SUPPORTED_AXES) + ".\n"
    "  - out_of_scope: true if the question is not about company financials.\n"
    "  - reason: one short sentence. Empty for in-scope inputs.\n\n"
    "Respond with ONLY a JSON object matching the schema. No prose, no markdown."
)

# The local 7B intermittently emits list fields as bare strings ("growth" instead
# of ["growth"]), which fails schema validation and gets swallowed into a refusal.
# Constrained decoding pins the output to the exact shape. Only the vLLM path uses
# it; Anthropic models are reliably compliant and their API rejects this param.
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "ClassifyOutput", "schema": ClassifyOutput.model_json_schema()},
}


def _parse_output(raw_text: str) -> ClassifyOutput:
    # The local 7B is usually compliant; Sonnet/Haiku are reliable. We strip a
    # ```json``` fence if the model added one.
    s = raw_text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
    return ClassifyOutput.model_validate_json(s.strip())


def classify(
    question: str, *, client: ChatClient, trace_ctx: TraceCtx | None = None
) -> dict[str, Any]:
    # Node contract: returns a partial state dict. Never raises on a domain
    # failure (bad parse, off-topic input); routes to refuse via out_of_scope.
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": question},
    ]
    chat_kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.0,
    }
    if client.backend == "anthropic":
        chat_kwargs["system"] = _SYSTEM
    else:
        # vLLM/openai chat format: system goes in the messages array.
        chat_kwargs["messages"] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": question},
        ]
        chat_kwargs["response_format"] = _RESPONSE_FORMAT

    try:
        resp = client.chat(**chat_kwargs)
        if trace_ctx is not None:
            trace_ctx.event("llm:classifier", **llm_trace_fields(client.model, resp))
        text = _extract_text(resp, client.backend)
        parsed = _parse_output(text)
    except Exception as e:
        return {
            "out_of_scope": True,
            "refusal_reason": f"classifier_failure: {type(e).__name__}",
        }

    return {
        "companies_raw": parsed.companies_raw,
        "axes": list(parsed.axes),
        "out_of_scope": parsed.out_of_scope,
        "refusal_reason": parsed.reason if parsed.out_of_scope else None,
    }


def _extract_text(resp: Any, backend: str) -> str:
    if backend == "anthropic":
        blocks = getattr(resp, "content", None) or []
        for b in blocks:
            t = getattr(b, "text", None)
            if t:
                return str(t)
        return ""
    # OpenAI-compatible (vLLM)
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    msg = getattr(choices[0], "message", None)
    return str(getattr(msg, "content", "") or "")
