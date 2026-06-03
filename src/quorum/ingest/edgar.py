from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# EDGAR's published limit is 10 req/sec across all callers from one IP.
# Stay under it; back-off on 429.
EDGAR_RATE_LIMIT_PER_SEC = 10
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
FILING_DOC_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodashes}/{primary_doc}"
)
ACCESSION_HEADER_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodashes}/{accession}-index.json"
)


@dataclass(frozen=True, slots=True)
class FilingRef:
    cik: str  # unpadded
    accession: str  # with dashes
    form: str
    primary_doc: str
    filing_date: str
    report_date: str


class RateLimiter:
    # Token-style gate: one slot every interval seconds. Thread-safe so concurrent
    # ingest workers stay under the global cap.
    def __init__(self, max_per_sec: int = EDGAR_RATE_LIMIT_PER_SEC) -> None:
        self._interval = 1.0 / max_per_sec
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._last + self._interval - now)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


class EdgarFetcher:
    # Filesystem-cached HTTP client over EDGAR. The cache is keyed by URL path,
    # so a re-run on a previously fetched CIK or filing performs zero network I/O.
    def __init__(
        self,
        *,
        user_agent: str,
        cache_dir: Path,
        rate_limiter: RateLimiter | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not user_agent:
            raise ValueError(
                "EDGAR fair-access policy requires a User-Agent with a contact address. "
                "Set EDGAR_USER_AGENT (or EDGAR_UA)."
            )
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = rate_limiter or RateLimiter()
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=httpx.Timeout(30.0, read=120.0),
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def _cache_path_for(self, url: str) -> Path:
        # Mirror the URL's path under cache_dir. Safe: no `..` allowed by URL parser.
        from urllib.parse import urlparse

        parsed = urlparse(url)
        rel = (parsed.netloc + parsed.path).replace("?", "_")
        return self.cache_dir / rel

    def _get(self, url: str) -> bytes:
        path = self._cache_path_for(url)
        if path.exists():
            return path.read_bytes()
        self.rate_limiter.acquire()
        resp = self._client.get(url)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        return resp.content

    def fetch_submissions(self, cik: str) -> dict[str, Any]:
        from quorum.config.companies import cik_padded

        url = SUBMISSIONS_URL.format(cik_padded=cik_padded(cik))
        result: dict[str, Any] = json.loads(self._get(url))
        return result

    def fetch_company_facts(self, cik: str) -> dict[str, Any]:
        from quorum.config.companies import cik_padded

        url = COMPANYFACTS_URL.format(cik_padded=cik_padded(cik))
        result: dict[str, Any] = json.loads(self._get(url))
        return result

    def list_filings(
        self,
        cik: str,
        *,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        max_per_form: dict[str, int] | None = None,
    ) -> list[FilingRef]:
        # Most recent filings first per EDGAR's submissions JSON ordering.
        submissions = self.fetch_submissions(cik)
        recent = submissions["filings"]["recent"]
        seen: dict[str, int] = dict.fromkeys(forms, 0)
        cap = max_per_form or {}
        out: list[FilingRef] = []
        for i, form in enumerate(recent["form"]):
            if form not in forms:
                continue
            limit = cap.get(form)
            if limit is not None and seen[form] >= limit:
                continue
            seen[form] += 1
            out.append(
                FilingRef(
                    cik=cik,
                    accession=recent["accessionNumber"][i],
                    form=form,
                    primary_doc=recent["primaryDocument"][i],
                    filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i],
                )
            )
        return out

    def fetch_primary_doc(self, filing: FilingRef) -> bytes:
        url = FILING_DOC_URL.format(
            cik_int=int(filing.cik),
            accession_nodashes=filing.accession.replace("-", ""),
            primary_doc=filing.primary_doc,
        )
        return self._get(url)

    def fetch_accession_header(self, filing: FilingRef) -> dict[str, Any]:
        url = ACCESSION_HEADER_URL.format(
            cik_int=int(filing.cik),
            accession_nodashes=filing.accession.replace("-", ""),
            accession=filing.accession,
        )
        result: dict[str, Any] = json.loads(self._get(url))
        return result
