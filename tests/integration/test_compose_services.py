from __future__ import annotations

import socket
import urllib.request
from urllib.error import URLError

import psycopg
import pytest

pytestmark = pytest.mark.integration


def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def test_postgres_ready(postgres_url: str) -> None:
    if not _tcp_reachable("localhost", 5432):
        pytest.skip("postgres not reachable; run `docker compose up -d postgres`")
    with psycopg.connect(postgres_url, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone() == (1,)


def test_trace_events_table_present(postgres_url: str) -> None:
    if not _tcp_reachable("localhost", 5432):
        pytest.skip("postgres not reachable; run `docker compose up -d postgres`")
    with psycopg.connect(postgres_url, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'trace_events' ORDER BY ordinal_position"
        )
        cols = [r[0] for r in cur.fetchall()]
    # Canonical schema lives in docs/ARCHITECTURE.md 4.2.
    expected = {
        "id",
        "request_id",
        "trace_id",
        "node_name",
        "attempt_number",
        "timestamp",
        "duration_ms",
        "input_shape",
        "output_shape",
        "tokens_in",
        "tokens_out",
        "cache_read_tokens",
        "cost_dollars_billed",
        "cost_dollars_effective",
        "error_kind",
        "error_reason",
    }
    assert expected.issubset(set(cols)), f"missing columns: {expected - set(cols)}"


def test_qdrant_ready(qdrant_url: str) -> None:
    if not _tcp_reachable("localhost", 6333):
        pytest.skip("qdrant not reachable; run `docker compose up -d qdrant`")
    for path in ("/readyz", "/healthz"):
        try:
            with urllib.request.urlopen(qdrant_url.rstrip("/") + path, timeout=3) as r:
                if r.status == 200:
                    return
        except URLError:
            continue
    pytest.fail("qdrant /readyz and /healthz both unreachable")
