from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_package_imports() -> None:
    # Smoke gate for Phase 1: the package and its declared modules import.
    # Catches broken __init__.py, missing deps, syntax errors at scaffold time.
    import quorum  # noqa: F401
    import quorum.cache.canonical  # noqa: F401
    import quorum.cache.embed_cache  # noqa: F401
    import quorum.cache.llm_cache  # noqa: F401
    import quorum.config.settings  # noqa: F401
    import quorum.trace.logger  # noqa: F401
    import quorum.trace.writer  # noqa: F401


@pytest.mark.smoke
def test_settings_load() -> None:
    from quorum.config.settings import Settings

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.pg_pool_min >= 1
    assert s.pg_pool_max >= s.pg_pool_min
    assert s.max_concurrent_requests >= 1
