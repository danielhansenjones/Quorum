from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import psycopg
from psycopg_pool import ConnectionPool

ErrorKind = Literal["none", "transient", "terminal"]


@dataclass(slots=True)
class TraceEvent:
    request_id: UUID
    trace_id: UUID
    node_name: str
    attempt_number: int = 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int | None = None
    input_shape: dict[str, Any] | None = None
    output_shape: dict[str, Any] | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    cost_dollars_billed: float = 0.0
    cost_dollars_effective: float = 0.0
    error_kind: ErrorKind = "none"
    error_reason: str | None = None


_INSERT = """
INSERT INTO trace_events (
  request_id, trace_id, node_name, attempt_number, "timestamp",
  duration_ms, input_shape, output_shape,
  tokens_in, tokens_out, cache_read_tokens,
  cost_dollars_billed, cost_dollars_effective,
  error_kind, error_reason
) VALUES (
  %s, %s, %s, %s, %s,
  %s, %s, %s,
  %s, %s, %s,
  %s, %s,
  %s, %s
)
"""


class TraceWriter:
    # Thin wrapper around a psycopg ConnectionPool. write_event acquires a
    # connection per call, which is fine at the 100 events/sec gate; the pool
    # is sized for fan-out (settings.pg_pool_max).
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def write_event(self, event: TraceEvent) -> None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                _INSERT,
                (
                    str(event.request_id),
                    str(event.trace_id),
                    event.node_name,
                    event.attempt_number,
                    event.timestamp,
                    event.duration_ms,
                    json.dumps(event.input_shape) if event.input_shape is not None else None,
                    json.dumps(event.output_shape) if event.output_shape is not None else None,
                    event.tokens_in,
                    event.tokens_out,
                    event.cache_read_tokens,
                    event.cost_dollars_billed,
                    event.cost_dollars_effective,
                    event.error_kind,
                    event.error_reason,
                ),
            )
            conn.commit()


@dataclass(slots=True)
class TraceCtx:
    # Per-request emission handle threaded into nodes. Tracing must never break
    # the graph, so every emit swallows errors (missing table, bad id, etc.).
    writer: TraceWriter | None
    request_id: str | None
    trace_id: str | None

    def event(
        self,
        node_name: str,
        *,
        duration_ms: int | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cache_read_tokens: int = 0,
        cost_dollars_billed: float = 0.0,
        cost_dollars_effective: float = 0.0,
        error_kind: ErrorKind = "none",
        error_reason: str | None = None,
        input_shape: dict[str, Any] | None = None,
    ) -> None:
        if self.writer is None or not self.request_id or not self.trace_id:
            return
        with contextlib.suppress(Exception):
            self.writer.write_event(
                TraceEvent(
                    request_id=UUID(self.request_id),
                    trace_id=UUID(self.trace_id),
                    node_name=node_name,
                    duration_ms=duration_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cache_read_tokens=cache_read_tokens,
                    cost_dollars_billed=cost_dollars_billed,
                    cost_dollars_effective=cost_dollars_effective,
                    error_kind=error_kind,
                    error_reason=error_reason,
                    input_shape=input_shape,
                )
            )


def open_pool(
    *,
    conninfo: str,
    min_size: int,
    max_size: int,
) -> ConnectionPool:
    # The Phase 1 gate enforces pg_pool_min >= (max_concurrent_requests * 4) + 5;
    # the assertion lives at boot in api.main, not here, so this stays a thin
    # constructor reusable by tests.
    pool = ConnectionPool(
        conninfo=conninfo,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": False},
        open=False,
    )
    pool.open(wait=True, timeout=10.0)
    return pool


def ensure_connectable(conninfo: str) -> None:
    # Sync probe for /ready endpoint and CI smoke. One-shot connection, no pool.
    with psycopg.connect(conninfo, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        cur.fetchone()
