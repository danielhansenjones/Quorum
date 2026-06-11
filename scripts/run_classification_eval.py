from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from quorum.config.settings import get_settings
from quorum.eval.classification import compare_to_baseline, run_classification_eval
from quorum.eval.runner import load_gold
from quorum.models.router import get_client

DEFAULT_GOLD = Path("eval/datasets/v1/gold.yaml")
DEFAULT_BASELINE = Path("eval/baselines/classification_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 10f/10g: classifier axis-accuracy + refusal eval over the gold set"
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Compare against this baseline and exit non-zero on a >tolerance drop.",
    )
    parser.add_argument("--no-baseline", action="store_true", help="Skip the regression gate.")
    args = parser.parse_args()

    settings = get_settings()
    if settings.vllm_url is None and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set VLLM_URL or run via ./secret-run for the Haiku classifier.", flush=True)
        return 2

    classifier_client = get_client("classifier", vllm_url=settings.vllm_url)
    cases = load_gold(args.gold)
    result = run_classification_eval(cases, classifier_client=classifier_client)
    print(json.dumps(result, indent=2), flush=True)

    if not args.no_baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text())
        gate = compare_to_baseline(result, baseline)
        if gate["regressions"]:
            print(f"\nREGRESSION (tolerance {gate['tolerance']}):", flush=True)
            for r in gate["regressions"]:
                print(
                    f"  {r['metric']}: {r['baseline']:.3f} -> {r['current']:.3f} "
                    f"(drop {r['drop']:.3f})",
                    flush=True,
                )
            return 1
        print(f"\nbaseline OK (no metric dropped > {gate['tolerance']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
