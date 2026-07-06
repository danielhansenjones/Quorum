from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from quorum.config.settings import get_settings
from quorum.eval.judges import (
    _FAITHFULNESS_SYSTEM,
    _QUALITY_DIMENSIONS,
    _QUALITY_SYSTEM,
)
from quorum.tools.filing_section import FilingSectionNotFound, get_filing_section

DEFAULT_GLOB = "eval/results/campaign-*"
DEFAULT_OUT = Path("eval/datasets/judge_sft")


def _quality_example(d: dict) -> dict | None:
    q = d.get("scores", {}).get("quality") or {}
    if "judge_error" in q:
        return None
    label = {dim: q[dim] for dim in _QUALITY_DIMENSIONS if dim in q}
    if len(label) != len(_QUALITY_DIMENSIONS):
        return None
    label["notes"] = str(q.get("notes", ""))
    return {
        "messages": [
            {"role": "system", "content": _QUALITY_SYSTEM},
            {"role": "user", "content": f"REPORT:\n{(d.get('report') or '')[:6000]}"},
            {"role": "assistant", "content": json.dumps(label)},
        ],
        "task": "quality",
    }


def _faithfulness_examples(d: dict, qdrant: QdrantClient) -> tuple[list[dict], int]:
    # Only Sonnet-judged claims carry a distillable label; quant citations are
    # scored deterministically and excluded. Join claim text back to its qual
    # citation to recover the section the judge actually read.
    quals = {c["claim"]: c for c in (d.get("citations") or []) if c.get("kind") == "qual"}
    out: list[dict] = []
    missing = 0
    for cl in d.get("scores", {}).get("faithfulness", {}).get("claims") or []:
        if cl.get("judge") != "sonnet":
            continue
        cit = quals.get(cl["claim"])
        if cit is None:
            continue
        try:
            section = get_filing_section(
                qdrant, ticker=cit["ticker"], accession=cit["accession"], section=cit["section"]
            )
        except FilingSectionNotFound:
            missing += 1
            continue
        user = f"Claim: {cl['claim']}\n\nSection excerpt (first 4000 chars):\n{section.text[:4000]}"
        label = {"score": cl["score"], "reason": str(cl.get("reason", ""))}
        out.append(
            {
                "messages": [
                    {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps(label)},
                ],
                "task": "faithfulness",
            }
        )
    return out, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build QLoRA SFT data: distill the Sonnet judge's stored verdicts into "
        "(judge-prompt -> verdict) pairs. Split by base case_id to prevent arm leakage."
    )
    parser.add_argument("--glob", default=DEFAULT_GLOB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-every", type=int, default=5, help="1-in-N case_ids held out for val.")
    args = parser.parse_args()

    settings = get_settings()
    qdrant = QdrantClient(url=settings.qdrant_url)

    case_files = sorted(
        p
        for p in Path().glob(f"{args.glob}/*.json")
        if p.name not in ("cost_report.json", "summary.json")
    )
    if not case_files:
        print(f"no case files under {args.glob}", file=sys.stderr)
        return 1

    examples: list[dict] = []
    case_ids: list[str] = []
    n_missing = 0
    for i, p in enumerate(case_files, start=1):
        d = json.loads(p.read_text())
        case_id = d.get("case_id") or p.stem
        arm = p.parent.name
        print(f"[{i}/{len(case_files)}] {arm}/{case_id}", file=sys.stderr)
        if d.get("final_status") not in ("ok", "partial") or not (d.get("report") or "").strip():
            continue

        rows: list[dict] = []
        q = _quality_example(d)
        if q is not None:
            rows.append(q)
        faith, missing = _faithfulness_examples(d, qdrant)
        rows.extend(faith)
        n_missing += missing
        for r in rows:
            r["case_id"] = case_id
            r["arm"] = arm
            examples.append(r)
            case_ids.append(case_id)

    unique_ids = sorted(set(case_ids))
    val_ids = {cid for idx, cid in enumerate(unique_ids) if idx % args.val_every == 0}

    train = [e for e in examples if e["case_id"] not in val_ids]
    val = [e for e in examples if e["case_id"] in val_ids]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = args.out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.out / "val_case_ids.txt").write_text("".join(cid + "\n" for cid in sorted(val_ids)))

    def _counts(rows: list[dict]) -> str:
        f = sum(1 for r in rows if r["task"] == "faithfulness")
        qn = sum(1 for r in rows if r["task"] == "quality")
        return f"{len(rows)} ({f} faith, {qn} quality)"

    print(
        json.dumps(
            {
                "case_files": len(case_files),
                "unique_case_ids": len(unique_ids),
                "val_case_ids": len(val_ids),
                "section_lookups_missing": n_missing,
                "train": _counts(train),
                "val": _counts(val),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
