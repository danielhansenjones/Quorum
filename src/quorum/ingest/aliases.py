from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from psycopg_pool import ConnectionPool

DEFAULT_KEY = "default"
PER_TICKER_KEY = "per_ticker"
DEFAULT_TICKER_TOKEN = "_default_"


def load_aliases_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def expand_aliases(
    aliases: dict[str, Any],
) -> list[tuple[str, str, int, str]]:
    # (axis_metric_key, ticker_or_default, ordering, concept) rows.
    rows: list[tuple[str, str, int, str]] = []
    for axis_metric_key, body in aliases.items():
        default_chain: list[str] = body.get(DEFAULT_KEY, []) or []
        for i, concept in enumerate(default_chain):
            rows.append((axis_metric_key, DEFAULT_TICKER_TOKEN, i, concept))
        for ticker, chain in (body.get(PER_TICKER_KEY) or {}).items():
            for i, concept in enumerate(chain or []):
                rows.append((axis_metric_key, ticker, i, concept))
    return rows


_TRUNCATE = "TRUNCATE TABLE concept_aliases"
_INSERT = (
    "INSERT INTO concept_aliases (axis_metric_key, ticker_or_default, ordering, concept) "
    "VALUES (%s, %s, %s, %s)"
)


def populate_aliases(pool: ConnectionPool, aliases_yaml_path: Path) -> int:
    rows = expand_aliases(load_aliases_yaml(aliases_yaml_path))
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(_TRUNCATE)
        cur.executemany(_INSERT, rows)
        conn.commit()
    return len(rows)
