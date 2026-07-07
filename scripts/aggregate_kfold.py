from __future__ import annotations

import argparse
import json
from pathlib import Path

from quorum.eval.judge_correlation import correlate

# Pool the per-fold held-out pairs into one correlation. Each case_id is held out
# in exactly one fold, so the pooled set covers the whole corpus with a fold
# adapter that never trained on the case it scored. This is the cross-validated
# answer to "does the fine-tune agree with Sonnet", on ~32 questions instead of 7.


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate kfold_gate.sh fold outputs into a cross-validated judge correlation."
    )
    ap.add_argument("--kfold", type=int, required=True)
    ap.add_argument("--dir", type=Path, default=Path("eval/results/judge_correlation/kfold"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    q_pairs: list[tuple[float, float]] = []
    q_labels: list[str] = []
    f_pairs: list[tuple[float, float]] = []
    f_labels: list[str] = []
    for i in range(args.kfold):
        study = json.loads((args.dir / f"fold_{i}_study.json").read_text())
        for row in study["per_case"]:
            cid = row["case"]
            sq, lq = row.get("sonnet_quality"), row.get("local_quality")
            if sq is not None and lq is not None:
                q_pairs.append((sq, lq))
                q_labels.append(cid)
            sf, lf = row.get("sonnet_faith_qual_only"), row.get("local_faith_qual_only")
            if sf is not None and lf is not None:
                f_pairs.append((sf, lf))
                f_labels.append(cid)

    out = {
        "kfold": args.kfold,
        "note": "cross-validated: every case scored by an adapter that never trained on it",
        "quality": correlate(
            [s for s, _ in q_pairs], [x for _, x in q_pairs], labels=q_labels
        ),
        "faithfulness_qual_only": correlate(
            [s for s, _ in f_pairs], [x for _, x in f_pairs], labels=f_labels
        ),
    }
    dest = args.out or args.dir / "cv_summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
