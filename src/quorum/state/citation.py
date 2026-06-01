from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class QuantCitation(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["quant"] = "quant"
    claim: str
    ticker: str
    accession: str
    concept: str
    normalized: str | None = None
    value: str
    period: str
    unit: str


class QualCitation(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["qual"] = "qual"
    claim: str
    ticker: str
    accession: str
    section: str
    chunk_id: str
    quote: str


# Discriminated union on `kind`. Synthesis and the judge pattern-match on this.
Citation = Annotated[QuantCitation | QualCitation, Field(discriminator="kind")]
