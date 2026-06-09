from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil

import pytest
from pydantic import BaseModel

import quorum.state as state_pkg
from quorum.state import CHECKPOINT_MODELS
from quorum.state.axis import AxisResult, CompanyAxisFinding
from quorum.state.citation import QuantCitation


def _all_state_models() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for mod_info in pkgutil.iter_modules(state_pkg.__path__):
        mod = importlib.import_module(f"quorum.state.{mod_info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, BaseModel)
                and obj is not BaseModel
                and obj.__module__ == mod.__name__
            ):
                found.add((obj.__module__, obj.__name__))
    return found


def test_every_state_model_is_registered_for_checkpointing() -> None:
    # An explicit msgpack allowlist BLOCKS unregistered types on resume, so a
    # state model missing from CHECKPOINT_MODELS would silently break the
    # checkpointer. This guard fails loudly when a model is added without
    # registering it.
    registered = {(m.__module__, m.__name__) for m in CHECKPOINT_MODELS}
    missing = _all_state_models() - registered
    assert not missing, f"unregistered checkpoint models: {sorted(missing)}"


def test_no_duplicate_registrations() -> None:
    keys = [(m.__module__, m.__name__) for m in CHECKPOINT_MODELS]
    assert len(keys) == len(set(keys))


def _configured_serde() -> object:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINT_MODELS))


def test_state_types_roundtrip_without_unregistered_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = _configured_serde()
    obj = AxisResult(
        axis="profitability",
        mode="structured",
        per_company={
            "AAPL": CompanyAxisFinding(
                ticker="AAPL",
                assessment="strong [AAPL:Q0]",
                citations=[
                    QuantCitation(
                        claim="c",
                        ticker="AAPL",
                        accession="0000320193-24-000123",
                        concept="us-gaap:Revenues",
                        value="391035000000",
                        period="FY2024",
                        unit="USD",
                    )
                ],
            )
        },
        comparison="AAPL leads [AAPL:Q0]",
        grounding="weak",
    )
    with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus"):
        type_str, data = serde.dumps_typed(obj)  # type: ignore[attr-defined]
        back = serde.loads_typed((type_str, data))  # type: ignore[attr-defined]

    assert back == obj
    msgs = " ".join(r.getMessage().lower() for r in caplog.records)
    assert "unregistered" not in msgs
    assert "blocked" not in msgs


def test_unregistered_type_is_blocked_under_explicit_allowlist(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Proves the allowlist is doing real work: a type NOT in CHECKPOINT_MODELS is
    # blocked, which is exactly why the completeness guard above matters.
    serde = _configured_serde()

    class _Stray(BaseModel):
        x: int = 1

    with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus"):
        type_str, data = serde.dumps_typed(_Stray())  # type: ignore[attr-defined]
        serde.loads_typed((type_str, data))  # type: ignore[attr-defined]
    msgs = " ".join(r.getMessage().lower() for r in caplog.records)
    assert "blocked" in msgs
