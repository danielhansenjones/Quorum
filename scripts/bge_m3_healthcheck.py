#!/usr/bin/env python
"""BGE-M3 health check (Phase 2a gate).

Embeds a fixed finance probe and asserts output dimensions. Exits non-zero on
any deviation. Usable directly (`uv run python scripts/bge_m3_healthcheck.py`)
or imported as a sanity probe.
"""

from __future__ import annotations

import sys

from quorum.models.embed import DENSE_DIM, BGEM3Embedder, health_check


def main() -> int:
    embedder = BGEM3Embedder(device="cpu")
    try:
        health_check(embedder)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OK: BGE-M3 dense dim={DENSE_DIM} on cpu, sparse weights present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
