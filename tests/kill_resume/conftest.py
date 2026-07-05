from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from quorum.graph.axis_config import AXIS_CONCEPTS
from quorum.ingest.aliases import DEFAULT_TICKER_TOKEN

KR_DB_NAME = "quorum_kill_resume"
_ROOT = Path(__file__).resolve().parents[2]


def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def postgres_url() -> str:
    # Session-scoped override of the function-scoped fixture in tests/conftest.py:
    # kr_database is session-scoped and pytest forbids depending on a
    # narrower-scoped fixture.
    return os.environ.get(
        "POSTGRES_URL",
        "postgresql://quorum:quorum@localhost:5432/quorum",
    )


@pytest.fixture(scope="session")
def kr_database(postgres_url: str) -> str:
    if not _tcp_reachable("localhost", 5432):
        pytest.skip("postgres not reachable; run `docker compose up -d postgres`")

    with psycopg.connect(postgres_url, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {KR_DB_NAME} WITH (FORCE)")
        conn.execute(f"CREATE DATABASE {KR_DB_NAME}")

    test_url = postgres_url.rsplit("/", 1)[0] + f"/{KR_DB_NAME}"
    init_sql = (_ROOT / "postgres-init" / "01-init.sql").read_text()
    # growth reuses profitability.revenue; dedupe so the alias PK holds.
    keys = list(dict.fromkeys(AXIS_CONCEPTS["profitability"] + AXIS_CONCEPTS["growth"]))
    with psycopg.connect(test_url, autocommit=True) as conn:
        conn.execute(init_sql)
        with conn.cursor() as cur:
            for key in keys:
                cur.execute(
                    "INSERT INTO concept_aliases "
                    "(axis_metric_key, ticker_or_default, ordering, concept) "
                    "VALUES (%s, %s, %s, %s)",
                    (key, DEFAULT_TICKER_TOKEN, 0, "KR_" + key.replace(".", "_")),
                )
            for cik in ("320193", "789019"):
                for key in keys:
                    for year in range(2022, 2026):
                        cur.execute(
                            "INSERT INTO facts (cik, concept, period, unit, value, accession) "
                            "VALUES (%s, %s, %s, %s, %s, %s)",
                            (
                                cik,
                                "KR_" + key.replace(".", "_"),
                                f"FY{year}",
                                "USD",
                                1000 + (year - 2022),
                                "0000-kr-1",
                            ),
                        )
    return test_url


class Harness:
    def __init__(self, *, db_url: str, cache_dir: Path, work_dir: Path) -> None:
        self.db_url = db_url
        self.cache_dir = cache_dir
        self.request_id = str(uuid.uuid4())
        self.trace_id = str(uuid.uuid4())
        self.out_path = work_dir / "out.json"
        self.calls_log = work_dir / "calls.log"
        self.hook_dir = work_dir / "hooks"
        self.hook_dir.mkdir()

    def _env(self, *, mode: str, hook: str | None, dump: bool) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "KR_POSTGRES_URL": self.db_url,
                "KR_CACHE_DIR": str(self.cache_dir),
                "KR_OUT": str(self.out_path),
                "KR_REQUEST_ID": self.request_id,
                "KR_TRACE_ID": self.trace_id,
                "KR_MODE": mode,
                "KR_HOOK_DIR": str(self.hook_dir),
                "KR_CALLS_LOG": str(self.calls_log),
            }
        )
        # Stale values inherited from the parent environment would arm hooks or
        # dumps the test did not ask for.
        env.pop("KR_HOOK", None)
        env.pop("KR_DUMP_CHECKPOINTS", None)
        if hook is not None:
            env["KR_HOOK"] = hook
        if dump:
            env["KR_DUMP_CHECKPOINTS"] = "1"
        return env

    def run(
        self,
        mode: str = "start",
        hook: str | None = None,
        dump: bool = False,
        timeout: int = 120,
    ) -> dict[str, Any]:
        proc = subprocess.run(
            [sys.executable, "-m", "tests.kill_resume.runner"],
            cwd=_ROOT,
            env=self._env(mode=mode, hook=hook, dump=dump),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        assert proc.returncode == 0, f"runner failed (mode={mode}):\n{proc.stderr}"
        return json.loads(self.out_path.read_text())

    def start_until_hook(self, hook: str) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tests.kill_resume.runner"],
            cwd=_ROOT,
            env=self._env(mode="start", hook=hook, dump=False),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        hit = self.hook_dir / "hook.hit"
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if hit.exists():
                return proc
            if proc.poll() is not None:
                _, err = proc.communicate()
                pytest.fail(f"runner exited before hook {hook!r} fired:\n{err.decode()}")
            time.sleep(0.05)
        proc.kill()
        proc.wait()
        pytest.fail(f"hook {hook!r} did not fire within 60s")

    def sigkill(self, proc: subprocess.Popen[bytes]) -> None:
        # Popen.kill() is SIGKILL on POSIX: no cleanup, no atexit, no flush.
        proc.kill()
        proc.wait()

    def calls(self) -> list[str]:
        if not self.calls_log.exists():
            return []
        return self.calls_log.read_text().splitlines()

    def trace_counts(self) -> dict[str, int]:
        with psycopg.connect(self.db_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT node_name, COUNT(*) FROM trace_events "
                "WHERE request_id = %s GROUP BY node_name",
                (self.request_id,),
            )
            return {name: int(n) for name, n in cur.fetchall()}

    def resume(self, dump: bool = False) -> dict[str, Any]:
        return self.run(mode="resume", dump=dump)


class HarnessFactory:
    # One shared llm-cache dir per test: cache warmth surviving a SIGKILL and
    # crossing harnesses is the property under test.
    def __init__(self, *, db_url: str, base_dir: Path) -> None:
        self.db_url = db_url
        self.cache_dir = base_dir / "llm-cache"
        self.cache_dir.mkdir()
        self._base_dir = base_dir
        self._n = 0

    def new(self) -> Harness:
        self._n += 1
        work_dir = self._base_dir / f"h{self._n}"
        work_dir.mkdir()
        return Harness(db_url=self.db_url, cache_dir=self.cache_dir, work_dir=work_dir)


@pytest.fixture
def kr(kr_database: str, tmp_path: Path) -> HarnessFactory:
    return HarnessFactory(db_url=kr_database, base_dir=tmp_path)
