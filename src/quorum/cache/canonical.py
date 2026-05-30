from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    # Sorted dict keys at every level; list order preserved (semantically meaningful
    # for message arrays). Compact separators. UTF-8. The encoder is the contract
    # for cache-key stability, kill-resume tests, and the message-array hash used
    # by the critic node's multi-turn cache.
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_default,
    ).encode("utf-8")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _default(o: Any) -> Any:
    # Pydantic models, datetimes, pathlib.Paths, sets - the common offenders.
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"canonical_json: unsupported type {type(o).__name__}")
