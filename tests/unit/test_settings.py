from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from quorum.config.settings import Settings


@pytest.fixture
def isolated_env() -> Iterator[None]:
    saved = {
        k: os.environ.get(k)
        for k in [
            "POSTGRES_URL",
            "QDRANT_URL",
            "VLLM_URL",
            "PG_POOL_MIN",
            "PG_POOL_MAX",
            "MAX_CONCURRENT_REQUESTS",
            "MAX_CONCURRENT_AXES_PER_REQUEST",
        ]
    }
    for k in saved:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_pool_size_from_env(isolated_env: None) -> None:
    os.environ["PG_POOL_MIN"] = "12"
    os.environ["PG_POOL_MAX"] = "30"
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.pg_pool_min == 12
    assert s.pg_pool_max == 30


def test_pool_required_floor_for_fan_out(isolated_env: None) -> None:
    os.environ["MAX_CONCURRENT_REQUESTS"] = "4"
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    # The architecture formula: 4 requests x 4 axes per request = 16
    # axis-analyst connections in flight, plus 5 for checkpointer + traces.
    assert s.pg_pool_min_required == 21


def test_defaults_match_phase1_recommendations(isolated_env: None) -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.max_concurrent_requests == 4
    assert s.max_concurrent_axes_per_request == 4
