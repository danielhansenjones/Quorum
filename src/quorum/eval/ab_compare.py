from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Phase 12e / 13b. Arm-agnostic paired comparison of two eval run dirs by
# case_id. An "arm" is one frozen-graph run under a toggle combination
# (critic_enabled, later rebuttal_enabled / agentic_analyst); the campaign is a
# sequence of these comparisons (baseline vs +critic vs +rebuttal vs +agentic),
# each one a call here. Pure and file-based so it unit-tests without services;
# cost is paired in only when the caller supplies per-request cost maps (DB).

_METRICS = ("faithfulness", "quality", "incorporation", "latency")


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def _quality_mean(quality: dict[str, Any] | None) -> float | None:
    if not quality:
        return None
    ints = [v for v in quality.values() if isinstance(v, int)]
    return (sum(ints) / len(ints)) if ints else None


def summarize_rebuttals(report: str, rebuttals: list[dict[str, Any]]) -> dict[str, Any]:
    # Phase 13b. Of the claims sent back to the analyst: how many defended /
    # retracted / revised, and whether the final report reflects each outcome.
    # Reflected proxy (deterministic): a retracted claim is absent from the
    # report (dropped); a defended/revised claim is present (kept). A reworded
    # defense reads as not-reflected - conservative, documented.
    counts = {"defended": 0, "retracted": 0, "revised": 0}
    nr = _normalize(report)
    reflected = 0
    for rb in rebuttals:
        disp = str(rb.get("disposition", ""))
        if disp in counts:
            counts[disp] += 1
        claim = _normalize(str(rb.get("claim", "")))
        present = bool(claim) and claim in nr
        if disp == "retracted":
            reflected += int(not present)
        elif disp in ("defended", "revised"):
            reflected += int(present)
    n = len(rebuttals)
    return {
        "n": n,
        **counts,
        "reflected": reflected,
        "reflected_rate": (reflected / n) if n else None,
    }


def extract_case_metrics(case_json: dict[str, Any]) -> dict[str, Any]:
    scores = case_json.get("scores") or {}
    faith = (scores.get("faithfulness") or {}).get("mean_score")
    incorporation = (scores.get("incorporation") or {}).get("rate")
    return {
        "case_id": case_json.get("case_id"),
        "request_id": case_json.get("request_id"),
        "faithfulness": faith,
        "quality": _quality_mean(scores.get("quality")),
        "incorporation": incorporation,
        "latency": case_json.get("elapsed_s"),
        "report": case_json.get("report") or "",
        "rebuttals": case_json.get("rebuttals") or [],
    }


def load_run_dir(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(Path(path).glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cid = d.get("case_id")
        if cid:
            out[str(cid)] = extract_case_metrics(d)
    return out


def _paired_deltas(
    a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], metric: str
) -> list[float]:
    # Delta convention: arm_b - arm_a. Only cases present in both with a non-None
    # value on both sides contribute (a refusal has no faithfulness, etc.).
    deltas: list[float] = []
    for cid in sorted(set(a) & set(b)):
        va, vb = a[cid].get(metric), b[cid].get(metric)
        if va is not None and vb is not None:
            deltas.append(float(vb) - float(va))
    return deltas


def bootstrap_ci(
    deltas: list[float], *, confidence: float = 0.95, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    # Paired bootstrap on the per-case deltas. Fixed seed -> deterministic CI.
    import numpy as np
    from scipy import stats  # eval extra; lazy

    res = stats.bootstrap(
        (np.asarray(deltas, dtype=float),),
        statistic=np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        random_state=seed,
        method="percentile",
    )
    ci = res.confidence_interval
    return float(ci.low), float(ci.high)


def _delta_block(deltas: list[float]) -> dict[str, Any]:
    n = len(deltas)
    if n == 0:
        return {"n": 0, "mean_delta": None, "ci95": [None, None]}
    mean = sum(deltas) / n
    if n < 2:
        return {"n": n, "mean_delta": mean, "ci95": [mean, mean]}
    lo, hi = bootstrap_ci(deltas)
    return {"n": n, "mean_delta": mean, "ci95": [lo, hi]}


def _arm_rebuttal_summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = {"n": 0, "defended": 0, "retracted": 0, "revised": 0, "reflected": 0}
    for rec in records.values():
        s = summarize_rebuttals(str(rec.get("report", "")), rec.get("rebuttals") or [])
        for k in total:
            total[k] += int(s[k])
    return {**total, "reflected_rate": (total["reflected"] / total["n"]) if total["n"] else None}


def compare_runs(
    dir_a: str | Path,
    dir_b: str | Path,
    *,
    cost_a: dict[str, float] | None = None,
    cost_b: dict[str, float] | None = None,
) -> dict[str, Any]:
    a = load_run_dir(dir_a)
    b = load_run_dir(dir_b)
    paired = sorted(set(a) & set(b))
    metrics: dict[str, Any] = {m: _delta_block(_paired_deltas(a, b, m)) for m in _METRICS}
    if cost_a is not None and cost_b is not None:
        cost_deltas: list[float] = []
        for cid in paired:
            ca = cost_a.get(str(a[cid].get("request_id")))
            cb = cost_b.get(str(b[cid].get("request_id")))
            if ca is not None and cb is not None:
                cost_deltas.append(float(cb) - float(ca))
        metrics["cost"] = _delta_block(cost_deltas)
    return {
        "arm_a": str(dir_a),
        "arm_b": str(dir_b),
        "paired_cases": len(paired),
        "only_in_a": sorted(set(a) - set(b)),
        "only_in_b": sorted(set(b) - set(a)),
        "metrics": metrics,
        "rebuttal": {"arm_a": _arm_rebuttal_summary(a), "arm_b": _arm_rebuttal_summary(b)},
    }
