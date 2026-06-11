from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx


class Palette:
    def __init__(self, enabled: bool) -> None:
        def code(c: str) -> str:
            return c if enabled else ""

        self.R = code("\033[0m")
        self.DIM = code("\033[2m")
        self.BOLD = code("\033[1m")
        self.CY = code("\033[36m")
        self.GREEN = code("\033[32m")
        self.YEL = code("\033[33m")
        self.RED = code("\033[31m")
        self.MAG = code("\033[35m")


LABEL_W = 11


def _emit(pal: Palette, label: str, body: str, color: str | None = None) -> None:
    color = pal.CY if color is None else color
    print(f"  {color}{label:<{LABEL_W}}{pal.R} {body}")


def _truncate(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _fmt_args(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        parts.append(f"{k}={_truncate(str(v), 30)}")
    return " ".join(parts)


def _grounding_color(pal: Palette, grounding: str) -> str:
    if grounding == "ok":
        return pal.GREEN
    if grounding == "weak":
        return pal.YEL
    return pal.RED


def render_node(pal: Palette, node: str, detail: dict[str, Any]) -> None:
    if node == "classify":
        if detail.get("out_of_scope"):
            _emit(pal, "classify", f"{pal.YEL}out of scope{pal.R}")
            return
        axes = ", ".join(detail.get("axes", []))
        mentions = ", ".join(detail.get("companies_raw", []))
        _emit(pal, "classify", f"in-scope  axes=[{axes}]  mentions=[{mentions}]")
    elif node == "resolve":
        _emit(pal, "resolve", f"tickers=[{', '.join(detail.get('tickers', []))}]")
    elif node == "plan":
        tasks = detail.get("tasks", [])
        if detail.get("replan_count"):
            _emit(pal, "plan", f"{pal.DIM}re-plan #{detail['replan_count']}{pal.R}")
        _emit(
            pal,
            "plan",
            f"{len(tasks)} analyst task(s) -> fan-out: [{', '.join(tasks)}]   "
            f"budget={detail.get('remaining_steps')}",
        )
    elif node == "analyze_axis":
        for r in detail.get("results", []):
            gc = _grounding_color(pal, r["grounding"])
            _emit(
                pal,
                "analyze",
                f"{r['axis']:<16} done  grounding={gc}{r['grounding']}{pal.R}  "
                f"{r['citations']} cites",
            )
    elif node == "assess":
        _emit(pal, "assess", f"route -> {pal.BOLD}{detail.get('route')}{pal.R}")
    elif node == "critic":
        if detail.get("critique") is None:
            _emit(pal, "critic", f"{pal.DIM}unavailable (bypassed / timeout / failed){pal.R}")
            return
        _emit(
            pal,
            "critic",
            f"{pal.MAG}agent loop{pal.R}  turns={detail.get('turns_used')}  "
            f"status={detail.get('status')}  ({detail.get('duration_ms')} ms)",
        )
        for i, tc in enumerate(detail.get("tool_calls", []), 1):
            ok = f"{pal.GREEN}ok{pal.R}" if tc["ok"] else f"{pal.RED}err{pal.R}"
            print(
                f"    {pal.DIM}[{i}]{pal.R} {tc['tool']}({_fmt_args(tc['args'])})  "
                f"-> {_truncate(tc['result'], 48)}  [{ok}]"
            )
        for f in detail.get("flags", []):
            print(
                f"    {pal.YEL}FLAG{pal.R} [{f['axis']}] {f['flag']}: {_truncate(f['claim'], 64)}"
            )
            print(f"         {pal.DIM}reason: {_truncate(f['reason'], 78)}{pal.R}")
    elif node == "synthesize":
        _emit(
            pal,
            "synthesize",
            f"status={detail.get('status')}  {detail.get('citations')} citations",
        )
    elif node == "refuse":
        _emit(pal, "refuse", f"{pal.YEL}{detail.get('status')}{pal.R}")


def render_final(pal: Palette, data: dict[str, Any]) -> None:
    print(f"\n  {pal.DIM}{'-' * 72}{pal.R}")
    cits = data.get("citations") or []
    _emit(
        pal,
        "REPORT",
        f"status={data.get('status')}  {len(cits)} citations  request_id={data.get('request_id')}",
        color=pal.BOLD,
    )
    print()
    report = data.get("report") or "(no report)"
    for ln in report.splitlines():
        print(f"    {ln}")


def print_cost(pal: Palette, request_id: str) -> None:
    from quorum.config.settings import get_settings
    from quorum.eval.cost_report import cost_report_from_db
    from quorum.trace.writer import open_pool

    settings = get_settings()
    pool = open_pool(conninfo=settings.postgres_url, min_size=1, max_size=2)
    try:
        report: dict[str, Any] = {}
        # trace rows are written per node during the run; allow a brief lag.
        for _ in range(4):
            report = cost_report_from_db(pool, request_ids=[request_id])
            if report.get("per_node"):
                break
            time.sleep(0.5)
    finally:
        pool.close()

    totals = report.get("totals", {})
    per_node = report.get("per_node", {})
    print(f"\n  {pal.DIM}{'-' * 72}{pal.R}")
    _emit(
        pal,
        "COST",
        f"total ${totals.get('cost', 0.0):.4f}   "
        f"cache_read={totals.get('cache_read_fraction', 0.0):.0%}",
        color=pal.BOLD,
    )
    for node, v in per_node.items():
        print(f"    {node:<24} ${v['cost_total']:.4f}   in={v['tokens_in']} out={v['tokens_out']}")


def stream(url: str, question: str, max_replans: int, pal: Palette, step: float) -> str | None:
    payload = {"question": question, "max_replans": max_replans}
    request_id: str | None = None
    event_type: str | None = None
    data_lines: list[str] = []

    def dispatch(ev: str, raw: str) -> None:
        nonlocal request_id
        data = json.loads(raw)
        if ev == "node":
            node = data["node"]
            if node == "entry":
                return
            render_node(pal, node, data.get("detail", {}))
            if step:
                time.sleep(step)
        elif ev == "final":
            request_id = data.get("request_id")
            render_final(pal, data)
        elif ev == "error":
            print(f"  {pal.RED}error: {data.get('error')}: {data.get('detail')}{pal.R}")

    with httpx.stream("POST", f"{url}/compare", json=payload, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line == "":
                if data_lines:
                    dispatch(event_type or "message", "\n".join(data_lines))
                event_type, data_lines = None, []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())

    return request_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream a /compare run and render the agent trajectory live.",
        epilog=(
            "Start the server first (e.g. `uvicorn quorum.api.main:app`), then:\n"
            '  python scripts/demo.py "Compare AAPL and MSFT profitability"\n'
            "Use --step to pace the trajectory for a screen-share; warm the LLM "
            "cache first for a free, deterministic run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--max-replans", type=int, default=2)
    parser.add_argument(
        "--step",
        type=float,
        default=0.0,
        help="Seconds to pause after each node, for watchable pacing.",
    )
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--cost",
        action="store_true",
        help="Print the per-node cost breakdown after the run (needs Postgres).",
    )
    args = parser.parse_args()

    pal = Palette(enabled=not args.no_color and sys.stdout.isatty())
    print(f"\n  {pal.BOLD}quorum{pal.R} {pal.DIM}>{pal.R} {args.question}\n")
    request_id = stream(args.url, args.question, args.max_replans, pal, args.step)
    if args.cost and request_id:
        print_cost(pal, request_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
