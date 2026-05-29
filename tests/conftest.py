from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def postgres_url() -> str:
    return os.environ.get(
        "POSTGRES_URL",
        "postgresql://quorum:quorum@localhost:5432/quorum",
    )


@pytest.fixture
def qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", "http://localhost:6333")
