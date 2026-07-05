from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "demo.py"

_EVENTS: list[tuple[str, dict[str, object]]] = [
    ("node", {"node": "entry", "detail": {}}),
    (
        "node",
        {
            "node": "classify",
            "detail": {"axes": ["profitability"], "companies_raw": ["Apple", "Microsoft"]},
        },
    ),
    ("node", {"node": "resolve", "detail": {"tickers": ["AAPL", "MSFT"]}}),
    ("node", {"node": "plan", "detail": {"tasks": ["profitability"], "remaining_steps": 2}}),
    (
        "node",
        {
            "node": "analyze_axis",
            "detail": {"results": [{"axis": "profitability", "grounding": "ok", "citations": 4}]},
        },
    ),
    ("node", {"node": "assess", "detail": {"axes": 1, "weak": 0}}),
    # A successful critique arrives as flat fields with NO "critique" key;
    # the API only sends {"critique": None} for the bypassed path.
    (
        "node",
        {
            "node": "critic",
            "detail": {
                "status": "ok",
                "turns_used": 2,
                "duration_ms": 41,
                "tool_calls": [
                    {
                        "tool": "get_financial_concept",
                        "args": {"ticker": "AAPL"},
                        "ok": True,
                        "result": "value=93736000000",
                    }
                ],
                "flags": [
                    {
                        "axis": "profitability",
                        "flag": "unsupported",
                        "claim": "MSFT margin grew 31%",
                        "reason": "filing states 21%",
                    }
                ],
            },
        },
    ),
    ("node", {"node": "synthesize", "detail": {"status": "answered", "citations": 4}}),
    (
        "final",
        {
            "status": "answered",
            "citations": [1, 2, 3, 4],
            "request_id": "req-123",
            "report": "AAPL leads on margin.\nMSFT leads on growth.",
        },
    ),
]

_COST = {
    "totals": {"cost": 0.0123, "cache_read_fraction": 0.97},
    "per_node": {"analyze_axis": {"cost_total": 0.01, "tokens_in": 900, "tokens_out": 200}},
}


def _write_fixture(path: Path, *, with_cost: bool = True) -> None:
    lines = [{"event": "meta", "data": {"question": "Compare AAPL and MSFT profitability"}}]
    lines += [{"event": ev, "data": data} for ev, data in _EVENTS]
    if with_cost:
        lines.append({"event": "cost", "data": _COST})
    path.write_text("".join(json.dumps(ln) + "\n" for ln in lines), encoding="utf-8")


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_replay_renders_fixture_without_server(tmp_path: Path) -> None:
    fixture = tmp_path / "demo.jsonl"
    _write_fixture(fixture)
    proc = _run("--replay", str(fixture), "--no-color")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "(replay)" in out
    assert "Compare AAPL and MSFT profitability" in out
    assert "axes=[profitability]" in out
    assert "tickers=[AAPL, MSFT]" in out
    assert "grounding=ok" in out
    assert "0 weak" in out
    assert "agent loop  turns=2  status=ok" in out
    assert "get_financial_concept(ticker=AAPL)" in out
    assert "FLAG [profitability] unsupported: MSFT margin grew 31%" in out
    assert "unavailable" not in out
    assert "status=answered  4 citations  request_id=req-123" in out
    assert "MSFT leads on growth." in out
    assert "total $0.0123" in out
    assert "cache_read=97%" in out


def test_replay_renders_bypassed_critic_as_unavailable(tmp_path: Path) -> None:
    fixture = tmp_path / "demo.jsonl"
    lines = [
        {"event": "meta", "data": {"question": "q"}},
        {"event": "node", "data": {"node": "critic", "detail": {"critique": None}}},
    ]
    fixture.write_text("".join(json.dumps(ln) + "\n" for ln in lines), encoding="utf-8")
    proc = _run("--replay", str(fixture), "--no-color")
    assert proc.returncode == 0, proc.stderr
    assert "unavailable (bypassed / timeout / failed)" in proc.stdout


def test_replay_skips_entry_node(tmp_path: Path) -> None:
    fixture = tmp_path / "demo.jsonl"
    _write_fixture(fixture)
    proc = _run("--replay", str(fixture), "--no-color")
    assert "entry" not in proc.stdout


def test_replay_missing_fixture_fails_with_hint(tmp_path: Path) -> None:
    proc = _run("--replay", str(tmp_path / "absent.jsonl"))
    assert proc.returncode == 2
    assert "replay fixture not found" in proc.stderr
    assert "--record" in proc.stderr


def test_question_required_without_replay() -> None:
    proc = _run()
    assert proc.returncode == 2
    assert "question is required" in proc.stderr


def test_record_conflicts_with_replay(tmp_path: Path) -> None:
    proc = _run("--replay", "--record", str(tmp_path / "out.jsonl"))
    assert proc.returncode == 2


class _SSEHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev, data in _EVENTS:
            self.wfile.write(f"event: {ev}\ndata: {json.dumps(data)}\n\n".encode())

    def log_message(self, *args: object) -> None:
        pass


def test_record_then_replay_round_trip(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fixture = tmp_path / "recorded.jsonl"
    try:
        live = _run(
            "Compare AAPL and MSFT profitability",
            "--url",
            f"http://127.0.0.1:{server.server_address[1]}",
            "--record",
            str(fixture),
            "--no-color",
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert live.returncode == 0, live.stderr

    entries = [json.loads(ln) for ln in fixture.read_text(encoding="utf-8").splitlines()]
    assert entries[0] == {
        "data": {"question": "Compare AAPL and MSFT profitability"},
        "event": "meta",
    }
    assert [e["event"] for e in entries[1:]] == [ev for ev, _ in _EVENTS]

    replayed = _run("--replay", str(fixture), "--no-color")
    assert replayed.returncode == 0, replayed.stderr
    # The recorded fixture must reproduce the live render exactly, banner aside.
    live_body = live.stdout.split("\n", 2)[2]
    replay_body = replayed.stdout.split("\n", 2)[2]
    assert replay_body == live_body
