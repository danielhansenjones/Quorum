from __future__ import annotations

import argparse
import json

from quorum.config.settings import get_settings
from quorum.eval.cost_report import cost_report_from_db
from quorum.trace.writer import open_pool


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 10j: aggregate trace_events into per-request/per-node cost"
    )
    parser.add_argument(
        "--request-ids",
        type=str,
        default="",
        help="Comma-separated request_ids to scope the report; empty = all rows.",
    )
    args = parser.parse_args()

    settings = get_settings()
    pool = open_pool(conninfo=settings.postgres_url, min_size=1, max_size=4)
    try:
        ids = [s.strip() for s in args.request_ids.split(",") if s.strip()] or None
        report = cost_report_from_db(pool, request_ids=ids)
    finally:
        pool.close()
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
