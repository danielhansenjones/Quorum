from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    postgres_url: str = "postgresql://quorum:quorum@localhost:5432/quorum"
    qdrant_url: str = "http://localhost:6333"
    vllm_url: str | None = None

    anthropic_api_key: str | None = None
    edgar_user_agent: str = ""

    # Concurrency. The pool formula matches the Phase 1 gate:
    # min connections must be (max_concurrent_requests * 4) + 5 to survive
    # parallel axis analysts + checkpointer writes + trace writes per node.
    max_concurrent_requests: int = 4
    max_concurrent_axes_per_request: int = 4

    pg_pool_min: int = Field(default=10, ge=1)
    pg_pool_max: int = Field(default=50, ge=1)

    cache_dir: Path = Path("./data/cache")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pg_pool_min_required(self) -> int:
        return (self.max_concurrent_requests * 4) + 5


def get_settings() -> Settings:
    return Settings()
