# Phase 1.5 walking skeleton. Throwaway. AAPL + MSFT, profitability, dense-only,
# Qdrant in-process, no checkpointer, no cache, no critic, no eval.
# Production phases reimplement everything; see docs/quorum_build_checklist.md.

from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from anthropic import Anthropic
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# Populate os.environ from .env before any module-level env reads.
# Does not override existing env vars, so secret-run's ANTHROPIC_API_KEY wins.
load_dotenv()

COMPANIES = [
    {"ticker": "AAPL", "cik": "320193"},
    {"ticker": "MSFT", "cik": "789019"},
]
COLLECTION = "filings"
DENSE_DIM = 1024
# SEC's EDGAR fair-access policy requires an identifying User-Agent with a contact address;
# generic UAs get rate-limited or 403'd. No default - set EDGAR_UA in your environment.
EDGAR_UA = os.environ["EDGAR_UA"]
CACHE_DIR = Path(__file__).parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)
SONNET = "claude-sonnet-4-6"


def fetch_latest_10k(cik: str) -> tuple[str, str]:
    cik_padded = cik.zfill(10)
    cached = CACHE_DIR / f"submissions_{cik_padded}.json"
    if not cached.exists():
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        r = httpx.get(url, headers={"User-Agent": EDGAR_UA}, timeout=30)
        r.raise_for_status()
        cached.write_text(r.text)
    submissions = json.loads(cached.read_text())
    recent = submissions["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return recent["accessionNumber"][i], recent["primaryDocument"][i]
    raise RuntimeError(f"No 10-K found for CIK {cik}")


def fetch_filing_html(cik: str, accession: str, primary_doc: str) -> str:
    cached = CACHE_DIR / f"{cik}_{accession}.html"
    if not cached.exists():
        accession_nodashes = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession_nodashes}/{primary_doc}"
        )
        r = httpx.get(url, headers={"User-Agent": EDGAR_UA}, timeout=120)
        r.raise_for_status()
        cached.write_text(r.text)
    return cached.read_text()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)


def chunk_text(text: str, target_chars: int = 3000, overlap: int = 400) -> list[str]:
    # Character-window chunking; the real phase parses by Item boundary.
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + target_chars])
        if i + target_chars >= len(text):
            break
        i += target_chars - overlap
    return out


_model: BGEM3FlagModel | None = None


def embedder() -> BGEM3FlagModel:
    global _model
    if _model is None:
        # use_fp16 saves memory; dense-only for the spike, ColBERT MV is v2 per design.
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model


def embed_dense(texts: list[str]) -> list[list[float]]:
    out = embedder().encode(
        texts,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return [v.tolist() for v in out["dense_vecs"]]


def make_qdrant() -> QdrantClient:
    client = QdrantClient(":memory:")
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        )
    return client


def upsert_company(client: QdrantClient, ticker: str, chunks: list[str], vectors: list[list[float]]) -> None:
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"{ticker}:{i}")),
            vector=vec,
            payload={
                "ticker": ticker,
                "chunk_id": f"{ticker}:{i}",
                "text": text,
            },
        )
        for i, (text, vec) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION, points=points)


def search_for_ticker(client: QdrantClient, query_vec: list[float], ticker: str, k: int = 3):
    # qdrant-client >= 1.10 replaced .search() with .query_points(); the wire
    # protocol is the same, the Python API moved.
    resp = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        query_filter=Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]),
        limit=k,
    )
    return resp.points


def analyst(question: str, evidence: dict[str, list[dict]]) -> str:
    blocks: list[str] = []
    for ticker, chunks in evidence.items():
        for c in chunks:
            blocks.append(f"[{c['chunk_id']}]\n{c['text']}")
    prompt = (
        f"Question: {question}\n\n"
        "You are comparing two companies on the requested axis using only the evidence chunks below. "
        "Cite chunks inline using the [TICKER:N] tags shown. If evidence is thin or missing, say so explicitly. "
        "Do not invent figures.\n\n"
        "Evidence:\n\n" + "\n\n---\n\n".join(blocks)
    )
    client = Anthropic()
    msg = client.messages.create(
        model=SONNET,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


_qdrant: QdrantClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qdrant
    _qdrant = make_qdrant()
    for c in COMPANIES:
        accession, primary_doc = fetch_latest_10k(c["cik"])
        html = fetch_filing_html(c["cik"], accession, primary_doc)
        text = html_to_text(html)
        chunks = chunk_text(text)
        vectors = embed_dense(chunks)
        upsert_company(_qdrant, c["ticker"], chunks, vectors)
        print(f"[ingest] {c['ticker']}: {len(chunks)} chunks from {accession}")
    yield


app = FastAPI(lifespan=lifespan)


class CompareRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"ok": True, "collection": COLLECTION}


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    assert _qdrant is not None, "Qdrant not initialized; lifespan did not run"
    q_vec = embed_dense([req.question])[0]
    evidence: dict[str, list[dict]] = {}
    for c in COMPANIES:
        hits = search_for_ticker(_qdrant, q_vec, c["ticker"], k=3)
        evidence[c["ticker"]] = [
            {"chunk_id": h.payload["chunk_id"], "text": h.payload["text"], "score": h.score}
            for h in hits
        ]
    report = analyst(req.question, evidence)
    return {
        "report": report,
        "evidence": {
            t: [{"chunk_id": e["chunk_id"], "score": e["score"]} for e in chunks]
            for t, chunks in evidence.items()
        },
    }
