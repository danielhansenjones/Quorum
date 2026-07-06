from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

# 10-K canonical section names. Item numbers are globally unique across PARTs.
ITEM_TO_CANONICAL: dict[str, str] = {
    "1": "item_1_business",
    "1A": "item_1a_risk_factors",
    "1B": "item_1b_unresolved_staff_comments",
    "1C": "item_1c_cybersecurity",
    "2": "item_2_properties",
    "3": "item_3_legal_proceedings",
    "4": "item_4_mine_safety",
    "5": "item_5_market_for_common_equity",
    "6": "item_6_reserved",
    "7": "item_7_mda",
    "7A": "item_7a_market_risk",
    "8": "item_8_financial_statements",
    "9": "item_9_changes_in_accountants",
    "9A": "item_9a_controls_and_procedures",
    "9B": "item_9b_other_information",
    "9C": "item_9c_foreign_jurisdictions",
    "10": "item_10_directors_executive_officers",
    "11": "item_11_executive_compensation",
    "12": "item_12_security_ownership",
    "13": "item_13_related_transactions",
    "14": "item_14_principal_accountant_fees",
    "15": "item_15_exhibits",
    "16": "item_16_form_10k_summary",
}

# Position of each item in canonical order. A 10-K body presents every item once
# in this order, which is what lets us reject out-of-order cross-references.
_ITEM_RANK: dict[str, int] = {tok: i for i, tok in enumerate(ITEM_TO_CANONICAL)}

# 10-Q canonical section names keyed by (PART, item_token). 10-Qs reuse item
# numbers across PART I (Financial Information) and PART II (Other Information)
# with completely different semantics; a part-aware key disambiguates.
# Item 1A is kept as "item_1a_risk_factors" so the risk_factors axis filter
# works against both 10-K (Part I, Item 1A) and 10-Q (Part II, Item 1A).
ITEM_TO_CANONICAL_10Q: dict[tuple[str, str], str] = {
    ("1", "1"): "item_1_financial_statements",
    ("1", "2"): "item_2_mda",
    ("1", "3"): "item_3_market_risk",
    ("1", "4"): "item_4_controls_and_procedures",
    ("2", "1"): "item_1_legal_proceedings",
    ("2", "1A"): "item_1a_risk_factors",
    ("2", "2"): "item_2_unregistered_sales",
    ("2", "3"): "item_3_defaults_upon_senior_securities",
    ("2", "4"): "item_4_mine_safety",
    ("2", "5"): "item_5_other_information",
    ("2", "6"): "item_6_exhibits",
}

# Sections whose chunks should NOT be indexed for semantic retrieval. Item 8 is
# the financial statements + notes, which are tables of XBRL facts; we surface
# those through the Postgres facts table, not vector search.
SECTIONS_EXCLUDED_FROM_VECTOR_INDEX: frozenset[str] = frozenset(
    {"item_8_financial_statements", "item_1_financial_statements"}
)

_ITEM_HEADER = re.compile(
    r"^\s*Item\s+(\d{1,2}[A-Z]?)\s*[\.\:\-]?\s*", re.IGNORECASE | re.MULTILINE
)

# Roman numerals I-IV for PART headers. Longest alternatives first so III doesn't
# get partially matched as II.
_PART_HEADER = re.compile(r"^\s*PART\s+(IV|III|II|I)\b", re.IGNORECASE | re.MULTILINE)

_ROMAN_TO_ARABIC: dict[str, str] = {"I": "1", "II": "2", "III": "3", "IV": "4"}


@dataclass(frozen=True, slots=True)
class Section:
    name: str  # canonical snake_case
    text: str
    char_start: int
    char_end: int


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "noscript"]):
        tag.decompose()
    raw = soup.get_text(separator="\n", strip=False)
    # Normalize whitespace: collapse runs of spaces/tabs, keep newlines so Item
    # headers retain their line-start anchor.
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _title_is_caps(text: str, end: int) -> bool:
    # A real section header is followed by its title in ALL CAPS ("RISK FACTORS",
    # "BUSINESS"); an in-body cross-reference is followed by a Title-Case phrase
    # ("Risk Factors - Concentration section"). This only breaks ties between
    # candidate offsets for the same item, never filters, so a filer that prints
    # Title-Case headers falls back to positional selection unharmed.
    tail = text[end : end + 60]
    words = re.findall(r"[A-Za-z]+", tail)
    first = next((w for w in words if len(w) >= 2), "")
    return len(first) >= 3 and first.isupper()


def _is_toc_entry(
    matches: list[tuple[int, int, str]],
    idx: int,
    *,
    window: int = 2000,
    min_distinct: int = 12,
    body_gap: int = 500,
) -> bool:
    # A table-of-contents or exhibit-index line is both densely surrounded by other
    # item headers AND immediately chained to the next one. A real header that sits
    # right after the TOC is also dense-adjacent, but it starts a long body, so its
    # gap to the next header is large - that gap is what tells the two apart.
    start = matches[idx][0]
    distinct = len({m[2] for m in matches if start - window <= m[0] <= start + window})
    gap_next = matches[idx + 1][0] - start if idx + 1 < len(matches) else 1 << 30
    return distinct >= min_distinct and gap_next < body_gap


def _find_item_offsets_10k(text: str) -> list[tuple[str, str | None, int]]:
    matches = [
        (m.start(), m.end(), m.group(1).upper())
        for m in _ITEM_HEADER.finditer(text)
        if m.group(1).upper() in _ITEM_RANK
    ]
    by_item: dict[str, list[tuple[int, int]]] = {}
    for i, (s, e, tok) in enumerate(matches):
        if not _is_toc_entry(matches, i):
            by_item.setdefault(tok, []).append((s, e))
    # A 10-K presents each item once, in canonical order. Walk items in that order
    # and take the first surviving occurrence after the previous boundary, so an
    # out-of-order cross-reference (an "Item 1A" cited inside Item 1) is skipped.
    # Among in-order candidates prefer one whose title is ALL CAPS.
    chosen: list[tuple[str, str | None, int]] = []
    prev = -1
    for tok in ITEM_TO_CANONICAL:
        cands = [(s, e) for (s, e) in by_item.get(tok, []) if s > prev]
        if not cands:
            continue
        caps = [(s, e) for (s, e) in cands if _title_is_caps(text, e)]
        s, _e = caps[0] if caps else cands[0]
        chosen.append((tok, None, s))
        prev = s
    return chosen


def _chained_run_toc(
    items: list[tuple[int, int, str, str]], *, body_gap: int, run_len: int
) -> set[int]:
    # A table-of-contents / navigation block is a run of consecutive item headers
    # each chained to the next by a small gap; a real body section breaks the chain
    # with its text. A run of run_len or more is such a block. The run's final header
    # is kept: a real section that abuts the block (small gap in, long body out) is
    # that last link, so stripping it would starve the first body section.
    strip: set[int] = set()
    n = len(items)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][0] - items[j][0] < body_gap:
            j += 1
        if j - i + 1 >= run_len:
            strip.update(range(i, j))
        i = j + 1
    return strip


def _find_item_offsets_10q(text: str) -> list[tuple[str, str | None, int]]:
    raw: list[tuple[int, int, str, str]] = []
    for m in _PART_HEADER.finditer(text):
        arabic = _ROMAN_TO_ARABIC.get(m.group(1).upper())
        if arabic:
            raw.append((m.start(), m.start(), "part", arabic))
    for m in _ITEM_HEADER.finditer(text):
        raw.append((m.start(), m.end(), "item", m.group(1).upper()))
    raw.sort(key=lambda e: e[0])

    # An item's part is the nearest preceding PART header; items before any PART
    # header are malformed and dropped.
    items: list[tuple[int, int, str, str]] = []
    current_part: str | None = None
    for start, end, kind, value in raw:
        if kind == "part":
            current_part = value
        elif current_part is not None:
            items.append((start, end, current_part, value))

    toc = _chained_run_toc(items, body_gap=500, run_len=6)
    by_key: dict[tuple[str, str], list[int]] = {}
    for idx, (s, _e, part, tok) in enumerate(items):
        if idx not in toc and (part, tok) in ITEM_TO_CANONICAL_10Q:
            by_key.setdefault((part, tok), []).append(s)

    # A 10-Q presents PART I then PART II, each item once in canonical order. Walk
    # keys in that order taking the first surviving occurrence after the previous
    # boundary, so a TOC entry for an item absent from the body (mine safety in a
    # pharma filing) and an out-of-order in-body cross-reference are both skipped.
    chosen: list[tuple[str, str | None, int]] = []
    prev = -1
    for part, tok in ITEM_TO_CANONICAL_10Q:
        cands = [s for s in by_key.get((part, tok), []) if s > prev]
        if not cands:
            continue
        chosen.append((tok, part, cands[0]))
        prev = cands[0]
    return chosen


def find_item_offsets(text: str, *, form: str = "10-K") -> list[tuple[str, str | None, int]]:
    # Returns (item_token, part_label_or_None, start_offset) triples in document
    # order. The forms differ: a 10-K numbers items globally and presents each once
    # in canonical order (_find_item_offsets_10k), while a 10-Q reuses item numbers
    # across PART I and PART II so its key is (part, item) (_find_item_offsets_10q).
    if form == "10-K":
        return _find_item_offsets_10k(text)
    return _find_item_offsets_10q(text)


def segment_into_sections(text: str, *, form: str = "10-K") -> list[Section]:
    boundaries = find_item_offsets(text, form=form)
    if not boundaries:
        return []
    sections: list[Section] = []
    for i, (token, part, start) in enumerate(boundaries):
        end = boundaries[i + 1][2] if i + 1 < len(boundaries) else len(text)
        name: str | None
        if form == "10-Q":
            if part is None:
                # Item appearing before any PART header in a 10-Q is malformed;
                # skip rather than guess.
                continue
            name = ITEM_TO_CANONICAL_10Q.get((part, token))
        else:
            name = ITEM_TO_CANONICAL.get(token)
        if name is None:
            continue
        sections.append(
            Section(name=name, text=text[start:end].strip(), char_start=start, char_end=end)
        )
    return sections


def parse_filing_html(html: str | bytes, *, form: str = "10-K") -> list[Section]:
    text = html_to_text(html)
    return segment_into_sections(text, form=form)
