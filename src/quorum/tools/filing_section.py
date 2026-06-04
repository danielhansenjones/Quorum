from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from quorum.ingest.qdrant_writer import COLLECTION_NAME


@dataclass(frozen=True, slots=True)
class FilingSection:
    ticker: str
    accession: str
    section: str
    text: str
    chunk_ids: list[str]


class FilingSectionNotFound(Exception):
    pass


def _ordinal_from_chunk_id(chunk_id: str) -> int:
    # chunk_id format: accession:section:NNNN (Phase 3c).
    try:
        return int(chunk_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 0


def get_filing_section(
    client: QdrantClient,
    *,
    ticker: str,
    accession: str,
    section: str,
) -> FilingSection:
    # Phase 4d: stitch the chunks for one (ticker, accession, section) back into
    # the section text. Adjacent chunks share `overlap_tokens` of text; we don't
    # try to dedupe overlap since the LLM tolerates the redundancy and the
    # alternative is a token-level diff.
    flt = qm.Filter(
        must=[
            qm.FieldCondition(key="ticker", match=qm.MatchValue(value=ticker)),
            qm.FieldCondition(key="accession", match=qm.MatchValue(value=accession)),
            qm.FieldCondition(key="section", match=qm.MatchValue(value=section)),
        ]
    )
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=flt,
        limit=1024,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        raise FilingSectionNotFound(
            f"No chunks for ticker={ticker} accession={accession} section={section}"
        )
    sorted_points = sorted(
        points,
        key=lambda p: _ordinal_from_chunk_id(
            str(p.payload.get("chunk_id", "") if p.payload else "")
        ),
    )
    parts = [str(p.payload.get("text", "")) for p in sorted_points if p.payload]
    return FilingSection(
        ticker=ticker,
        accession=accession,
        section=section,
        text="\n\n".join(parts),
        chunk_ids=[str(p.payload.get("chunk_id", "")) for p in sorted_points if p.payload],
    )
