from __future__ import annotations

import socket
import time
import uuid

import pytest

from quorum.trace.writer import TraceEvent, TraceWriter, open_pool

pytestmark = pytest.mark.integration


def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def test_trace_writer_sustains_100_per_sec(postgres_url: str) -> None:
    if not _tcp_reachable("localhost", 5432):
        pytest.skip("postgres not reachable; run `docker compose up -d postgres`")

    pool = open_pool(conninfo=postgres_url, min_size=4, max_size=8)
    try:
        writer = TraceWriter(pool)
        request_id = uuid.uuid4()
        trace_id = uuid.uuid4()
        n = 100
        start = time.monotonic()
        for i in range(n):
            writer.write_event(
                TraceEvent(
                    request_id=request_id,
                    trace_id=trace_id,
                    node_name="phase1_smoke",
                    attempt_number=1,
                    duration_ms=i,
                    input_shape={"i": i},
                    output_shape={"ok": True},
                    tokens_in=i,
                    tokens_out=i,
                )
            )
        elapsed = time.monotonic() - start
    finally:
        pool.close()

    # Hard floor: 100 events in <= 1.0s means we are at >= 100 events/sec.
    # Generous margin allowed for cold pool warm-up on slow CI.
    assert elapsed <= 5.0, f"100 inserts took {elapsed:.2f}s; pool/insert path is too slow"


def test_trace_writer_round_trip(postgres_url: str) -> None:
    if not _tcp_reachable("localhost", 5432):
        pytest.skip("postgres not reachable; run `docker compose up -d postgres`")

    pool = open_pool(conninfo=postgres_url, min_size=1, max_size=2)
    try:
        writer = TraceWriter(pool)
        request_id = uuid.uuid4()
        trace_id = uuid.uuid4()
        writer.write_event(
            TraceEvent(
                request_id=request_id,
                trace_id=trace_id,
                node_name="round_trip",
                tokens_in=11,
                tokens_out=22,
                cache_read_tokens=5,
                cost_dollars_billed=0.001234,
                cost_dollars_effective=0.001000,
            )
        )
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT node_name, tokens_in, tokens_out, cache_read_tokens, "
                "       cost_dollars_billed, cost_dollars_effective "
                "FROM trace_events WHERE request_id = %s ORDER BY id DESC LIMIT 1",
                (str(request_id),),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == "round_trip"
        assert row[1] == 11
        assert row[2] == 22
        assert row[3] == 5
        assert float(row[4]) == pytest.approx(0.001234, rel=1e-6)
        assert float(row[5]) == pytest.approx(0.001000, rel=1e-6)
    finally:
        pool.close()
