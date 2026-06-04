from __future__ import annotations

from dataclasses import dataclass

import psycopg
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from quorum.config.companies import TICKER_BY_CIK
from quorum.ingest.qdrant_writer import COLLECTION_NAME


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    ticker: str
    cik: str
    facts_count: int
    chunks_count: int


@dataclass(frozen=True, slots=True)
class FilingSummary:
    ticker: str
    accession: str
    form: str
    fiscal_period: str
    filing_date: str
    chunk_count: int


def list_corpus(qdrant: QdrantClient, postgres_conninfo: str) -> list[CorpusEntry]:
    # Cross-check Postgres facts against Qdrant chunks. Catches partial ingest
    # where one half completed and the other failed (Phase 4e gate).
    facts_by_cik: dict[str, int] = {}
    with psycopg.connect(postgres_conninfo, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("SELECT cik, count(*) FROM facts GROUP BY cik")
        for cik, n in cur.fetchall():
            facts_by_cik[str(cik)] = int(n)

    chunks_by_ticker: dict[str, int] = {}
    next_offset = None
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=next_offset,
            with_payload=["ticker"],
            with_vectors=False,
        )
        for p in points:
            if not p.payload:
                continue
            t = str(p.payload.get("ticker", ""))
            chunks_by_ticker[t] = chunks_by_ticker.get(t, 0) + 1
        if next_offset is None:
            break

    out: list[CorpusEntry] = []
    seen_ciks = set(facts_by_cik.keys()) | {
        cik for cik, ticker in TICKER_BY_CIK.items() if ticker in chunks_by_ticker
    }
    for cik in sorted(seen_ciks):
        ticker = TICKER_BY_CIK.get(cik, "")
        out.append(
            CorpusEntry(
                ticker=ticker,
                cik=cik,
                facts_count=facts_by_cik.get(cik, 0),
                chunks_count=chunks_by_ticker.get(ticker, 0),
            )
        )
    return out


def list_filings(qdrant: QdrantClient, *, ticker: str | None = None) -> list[FilingSummary]:
    flt: qm.Filter | None = None
    if ticker:
        flt = qm.Filter(must=[qm.FieldCondition(key="ticker", match=qm.MatchValue(value=ticker))])
    seen: dict[tuple[str, str], FilingSummary] = {}
    counts: dict[tuple[str, str], int] = {}
    next_offset = None
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=flt,
            limit=1000,
            offset=next_offset,
            with_payload=["ticker", "accession", "form", "fiscal_period", "filing_date"],
            with_vectors=False,
        )
        for p in points:
            if not p.payload:
                continue
            t = str(p.payload.get("ticker", ""))
            acc = str(p.payload.get("accession", ""))
            key = (t, acc)
            counts[key] = counts.get(key, 0) + 1
            if key not in seen:
                seen[key] = FilingSummary(
                    ticker=t,
                    accession=acc,
                    form=str(p.payload.get("form", "")),
                    fiscal_period=str(p.payload.get("fiscal_period", "")),
                    filing_date=str(p.payload.get("filing_date", "")),
                    chunk_count=0,
                )
        if next_offset is None:
            break
    # Patch chunk_count into the immutable dataclass via rebuild.
    return [
        FilingSummary(
            ticker=v.ticker,
            accession=v.accession,
            form=v.form,
            fiscal_period=v.fiscal_period,
            filing_date=v.filing_date,
            chunk_count=counts[(v.ticker, v.accession)],
        )
        for v in seen.values()
    ]
