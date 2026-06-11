from __future__ import annotations

import argparse
import json
from pathlib import Path

from quorum.eval.ab_compare import compare_runs, load_run_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 12e/13b: paired comparison of two eval run dirs by case_id"
    )
    parser.add_argument("arm_a", type=Path, help="Baseline run dir (e.g. eval/runs/baseline)")
    parser.add_argument("arm_b", type=Path, help="Treatment run dir (e.g. eval/runs/+critic)")
    parser.add_argument(
        "--cost",
        action="store_true",
        help="Pair per-case cost from trace_events (needs Postgres + request_ids in the run JSONs).",
    )
    args = parser.parse_args()

    cost_a = cost_b = None
    if args.cost:
        from quorum.config.settings import get_settings
        from quorum.eval.cost_report import per_request_cost_from_db
        from quorum.trace.writer import open_pool

        def _ids(d: Path) -> list[str]:
            return [str(r["request_id"]) for r in load_run_dir(d).values() if r.get("request_id")]

        settings = get_settings()
        pool = open_pool(conninfo=settings.postgres_url, min_size=1, max_size=4)
        try:
            cost_a = per_request_cost_from_db(pool, request_ids=_ids(args.arm_a))
            cost_b = per_request_cost_from_db(pool, request_ids=_ids(args.arm_b))
        finally:
            pool.close()

    report = compare_runs(args.arm_a, args.arm_b, cost_a=cost_a, cost_b=cost_b)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
