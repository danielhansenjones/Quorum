from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches

from quorum.config.companies import COMPANIES


@dataclass(frozen=True, slots=True)
class ResolvedCompany:
    ticker: str
    cik: str
    name: str


_NOT_FOUND_SENTINEL = "not_found"


def _normalize(s: str) -> str:
    return s.strip().lower().replace(",", "").replace(".", "")


def _name_aliases(name: str) -> set[str]:
    # Strip a leading "the " and common corporate suffixes so "coca-cola",
    # "the coca-cola company", and "coca-cola company" all match KO.
    base = _normalize(name)
    bases = {base}
    if base.startswith("the "):
        bases.add(base[len("the ") :])
    aliases: set[str] = set()
    for b in bases:
        aliases.add(b)
        # Longer connector-suffixes first so "eli lilly and company" yields the
        # marketing name "eli lilly", not the dangling "eli lilly and".
        for suffix in (
            " and company",
            " & company",
            " inc",
            " corporation",
            " corp",
            " company",
            " co",
            " ltd",
            " plc",
        ):
            if b.endswith(suffix):
                aliases.add(b[: -len(suffix)].rstrip())
        # First token is often the marketing name (e.g. "procter"); skip the
        # "the" article so it never becomes a bare alias.
        first = b.split(" ")[0]
        if first != "the":
            aliases.add(first)
    return aliases


def resolve_company(query: str) -> ResolvedCompany | None:
    # Tool contract (Phase 4a): exact CIK / ticker / name match wins. Fuzzy
    # name match is allowed only when there's exactly one close candidate;
    # ambiguity returns not_found rather than guessing.
    if not query:
        return None
    q = _normalize(query)
    if not q:
        return None

    # Exact ticker match
    for c in COMPANIES:
        if q == c.ticker.lower():
            return ResolvedCompany(c.ticker, c.cik, c.name)

    # Exact CIK match (accept padded or unpadded)
    q_digits = "".join(ch for ch in q if ch.isdigit())
    if q_digits:
        for c in COMPANIES:
            if q_digits == c.cik or q_digits == c.cik.zfill(10):
                return ResolvedCompany(c.ticker, c.cik, c.name)

    # Name alias match
    matches: list[ResolvedCompany] = []
    for c in COMPANIES:
        if q in _name_aliases(c.name):
            matches.append(ResolvedCompany(c.ticker, c.cik, c.name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return None  # ambiguous

    # Fuzzy: difflib close matches against full and aliased names. Single best
    # candidate above the cutoff wins; multiple candidates -> not_found.
    pool: list[tuple[str, ResolvedCompany]] = []
    for c in COMPANIES:
        rc = ResolvedCompany(c.ticker, c.cik, c.name)
        pool.append((_normalize(c.name), rc))
        for alias in _name_aliases(c.name):
            pool.append((alias, rc))
    fuzzy = get_close_matches(q, [k for k, _ in pool], n=2, cutoff=0.85)
    if len(fuzzy) == 1:
        for k, rc in pool:
            if k == fuzzy[0]:
                return rc
    return None
