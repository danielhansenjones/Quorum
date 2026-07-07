from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil, floor
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import OpenAI

from quorum.eval.judges import _QUALITY_SYSTEM
from quorum.eval.runner import _git_provenance
from quorum.graph.nodes.classify import _RESPONSE_FORMAT, _SYSTEM, _parse_output
from quorum.models.router import DEFAULT_VLLM_MODEL

DEFAULT_VLLM = "http://localhost:8001/v1"
DEFAULT_OUT = Path("eval/results/serving-v1")
DEFAULT_GOLD = Path("eval/datasets/v1/gold.yaml")
DEFAULT_JUDGE_REPORTS = Path("eval/results/campaign-critic")
DEFAULT_LEVELS = (1, 2, 4, 8, 16, 32)
WARMUP = 2


@dataclass(slots=True)
class Workload:
    name: str
    messages: list[list[dict[str, Any]]]
    max_tokens: int
    response_format: dict[str, Any] | None = None


@dataclass(slots=True)
class ReqResult:
    ttft_s: float
    latency_s: float
    completion_tokens: int | None
    text: str


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = floor(k), ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _classifier_messages(question: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": question}]


def _judge_messages(report: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": _QUALITY_SYSTEM},
        {"role": "user", "content": f"REPORT:\n{report[:6000]}"},
    ]


def _load_classifier_prompts(gold: Path) -> list[list[dict[str, Any]]]:
    import yaml

    cases = yaml.safe_load(gold.read_text())["cases"]
    return [_classifier_messages(c["question"]) for c in cases if c.get("question")]


def _load_judge_prompts(run_dir: Path) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    for p in sorted(run_dir.glob("*.json")):
        if p.name in ("summary.json", "cost_report.json"):
            continue
        report = (json.loads(p.read_text()).get("report") or "").strip()
        if report:
            out.append(_judge_messages(report))
    return out


def _run_one(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    response_format: dict[str, Any] | None,
) -> ReqResult:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    t0 = perf_counter()
    ttft: float | None = None
    parts: list[str] = []
    completion_tokens: int | None = None
    for chunk in client.chat.completions.create(**kwargs):
        # vLLM sends a final usage-only chunk with an empty choices list.
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                if ttft is None:
                    ttft = perf_counter() - t0
                parts.append(content)
        if chunk.usage is not None:
            completion_tokens = chunk.usage.completion_tokens
    total = perf_counter() - t0
    return ReqResult(
        ttft_s=ttft if ttft is not None else total,
        latency_s=total,
        completion_tokens=completion_tokens,
        text="".join(parts),
    )


def _run_level(
    client: OpenAI, model: str, workload: Workload, concurrency: int, n_requests: int
) -> dict[str, Any]:
    prompts = workload.messages
    tasks = [prompts[i % len(prompts)] for i in range(n_requests)]
    results: list[ReqResult] = []
    wall0 = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(_run_one, client, model, msgs, workload.max_tokens, workload.response_format)
            for msgs in tasks
        ]
        for f in as_completed(futs):
            results.append(f.result())
    wall = perf_counter() - wall0

    ttfts = [r.ttft_s for r in results]
    lats = [r.latency_s for r in results]
    decode_rates = [
        r.completion_tokens / (r.latency_s - r.ttft_s)
        for r in results
        if r.completion_tokens and r.latency_s > r.ttft_s
    ]
    total_out = sum(r.completion_tokens or 0 for r in results)
    return {
        "concurrency": concurrency,
        "n_requests": len(results),
        "wall_s": round(wall, 4),
        "ttft_s": {"p50": _round(_pct(ttfts, 0.5)), "p95": _round(_pct(ttfts, 0.95))},
        "latency_s": {"p50": _round(_pct(lats, 0.5)), "p95": _round(_pct(lats, 0.95))},
        "output_tokens_per_s_per_req": {
            "p50": _round(_pct(decode_rates, 0.5)),
            "p95": _round(_pct(decode_rates, 0.95)),
        },
        "system_tokens_per_s": _round(total_out / wall if wall else None),
        "total_output_tokens": total_out,
    }


def _round(x: float | None, n: int = 3) -> float | None:
    return round(x, n) if x is not None else None


def _warmup(client: OpenAI, model: str, workload: Workload) -> None:
    for msgs in workload.messages[:WARMUP]:
        _run_one(client, model, msgs, workload.max_tokens, workload.response_format)


def _sweep(
    client: OpenAI, model: str, workload: Workload, levels: tuple[int, ...], reqs: int
) -> list[dict[str, Any]]:
    print(f"[warmup] {workload.name}", file=sys.stderr)
    _warmup(client, model, workload)
    rows = []
    for c in levels:
        n = max(reqs, c)  # at least one request per in-flight slot
        print(f"[sweep] {workload.name} c={c} n={n}", file=sys.stderr, flush=True)
        rows.append(_run_level(client, model, workload, c, n))
    return rows


def _served_models(client: OpenAI) -> list[str]:
    return [m.id for m in client.models.list().data]


def exp_concurrency(
    client: OpenAI, model: str, classifier: Workload, judge: Workload, levels, reqs
) -> dict[str, Any]:
    return {
        "classifier": _sweep(client, model, classifier, levels, reqs),
        "judge": _sweep(client, model, judge, levels, reqs),
    }


def exp_guided(
    client: OpenAI, model: str, classifier_prompts, concurrency: int, reqs: int
) -> dict[str, Any]:
    out = {}
    for label, rf in (("constrained", _RESPONSE_FORMAT), ("free", None)):
        wl = Workload(f"classifier-{label}", classifier_prompts, max_tokens=256, response_format=rf)
        print(f"[guided] {label} c={concurrency}", file=sys.stderr, flush=True)
        _warmup(client, model, wl)
        level = _run_level(client, model, wl, concurrency, max(reqs, concurrency))
        valid = _valid_fraction(client, model, classifier_prompts, rf)
        out[label] = {**level, "schema_valid_fraction": valid}
    return out


def _valid_fraction(
    client: OpenAI, model: str, prompts, response_format: dict[str, Any] | None
) -> float | None:
    # Ties to gotcha 7: free decode on the 7B emits list fields as bare strings
    # that fail schema validation. Measured, not asserted.
    ok = 0
    n = min(len(prompts), 24)
    for msgs in prompts[:n]:
        r = _run_one(client, model, msgs, 256, response_format)
        try:
            _parse_output(r.text)
            ok += 1
        except Exception:  # noqa: BLE001
            pass
    return round(ok / n, 3) if n else None


def exp_lora(
    client: OpenAI,
    base_model: str,
    lora_model: str,
    judge: Workload,
    concurrency: int,
    reqs: int,
    served: list[str],
) -> dict[str, Any]:
    if lora_model not in served:
        print(
            f"[lora] adapter '{lora_model}' not served (have {served}); "
            f"start vLLM with docker-compose.lora.yml. Skipping.",
            file=sys.stderr,
        )
        return {"skipped": f"adapter {lora_model} not served"}
    out = {}
    for label, model in (("base", base_model), ("adapter", lora_model)):
        print(f"[lora] {label}={model} c={concurrency}", file=sys.stderr, flush=True)
        _warmup(client, model, judge)
        out[label] = _run_level(client, model, judge, concurrency, max(reqs, concurrency))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serving benchmark: concurrency sweep, guided-decoding overhead, LoRA cost, "
        "on the real classifier and judge workloads against the local vLLM box."
    )
    parser.add_argument(
        "--experiment",
        choices=["concurrency", "guided", "lora", "all"],
        default="all",
    )
    parser.add_argument("--vllm-url", type=str, default=DEFAULT_VLLM)
    parser.add_argument("--vllm-model", type=str, default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--lora-model", type=str, default="judge-qlora")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--judge-reports", type=Path, default=DEFAULT_JUDGE_REPORTS)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Concurrency levels for the sweep.",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=32,
        help="Requests per concurrency level. Long judge gens at low concurrency dominate wall time.",
    )
    parser.add_argument(
        "--fixed-concurrency",
        type=int,
        default=8,
        help="Concurrency for the guided and lora single-point comparisons.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    client = OpenAI(base_url=args.vllm_url, api_key="not-used")
    try:
        served = _served_models(client)
    except Exception as e:  # noqa: BLE001
        print(f"vLLM not reachable at {args.vllm_url}: {e}", file=sys.stderr)
        return 2

    classifier_prompts = _load_classifier_prompts(args.gold)
    judge_prompts = _load_judge_prompts(args.judge_reports)
    classifier = Workload("classifier", classifier_prompts, 256, response_format=_RESPONSE_FORMAT)
    judge = Workload("judge", judge_prompts, 700)
    levels = tuple(args.levels)
    reqs = args.requests_per_level

    experiments: dict[str, Any] = {}
    want = args.experiment
    if want in ("concurrency", "all"):
        experiments["concurrency"] = exp_concurrency(
            client, args.vllm_model, classifier, judge, levels, reqs
        )
    if want in ("guided", "all"):
        experiments["guided_decoding"] = exp_guided(
            client, args.vllm_model, classifier_prompts, args.fixed_concurrency, reqs
        )
    if want in ("lora", "all"):
        experiments["lora"] = exp_lora(
            client, args.vllm_model, args.lora_model, judge, args.fixed_concurrency, reqs, served
        )

    summary = {
        "config": {
            "vllm_url": args.vllm_url,
            "vllm_model": args.vllm_model,
            "lora_model": args.lora_model,
            "served_models": served,
            "levels": list(levels),
            "requests_per_level": reqs,
            "fixed_concurrency": args.fixed_concurrency,
            "n_classifier_prompts": len(classifier_prompts),
            "n_judge_prompts": len(judge_prompts),
            "report_truncation_chars": 6000,
        },
        "provenance": {
            **_git_provenance(),
            "started_at": datetime.now(UTC).isoformat(),
        },
        "experiments": experiments,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out / 'summary.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
