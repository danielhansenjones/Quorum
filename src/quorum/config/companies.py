from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Company:
    ticker: str
    cik: str
    name: str
    sector: str  # "BigTech" | "Staples" | "Pharma"


# Locked in docs/decisions.md #2. Twelve names across three sectors.
COMPANIES: tuple[Company, ...] = (
    Company("AAPL", "320193", "Apple Inc.", "BigTech"),
    Company("MSFT", "789019", "Microsoft Corporation", "BigTech"),
    Company("GOOGL", "1652044", "Alphabet Inc.", "BigTech"),
    Company("META", "1326801", "Meta Platforms, Inc.", "BigTech"),
    Company("PG", "80424", "The Procter & Gamble Company", "Staples"),
    Company("KO", "21344", "The Coca-Cola Company", "Staples"),
    Company("PEP", "77476", "PepsiCo, Inc.", "Staples"),
    Company("COST", "909832", "Costco Wholesale Corporation", "Staples"),
    Company("JNJ", "200406", "Johnson & Johnson", "Pharma"),
    Company("PFE", "78003", "Pfizer Inc.", "Pharma"),
    Company("MRK", "310158", "Merck & Co., Inc.", "Pharma"),
    Company("LLY", "59478", "Eli Lilly and Company", "Pharma"),
)

TICKER_BY_CIK: dict[str, str] = {c.cik: c.ticker for c in COMPANIES}
CIK_BY_TICKER: dict[str, str] = {c.ticker: c.cik for c in COMPANIES}


def cik_padded(cik: str) -> str:
    return cik.zfill(10)
