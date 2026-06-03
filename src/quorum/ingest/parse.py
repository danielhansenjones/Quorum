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


def find_item_offsets(text: str, *, form: str = "10-K") -> list[tuple[str, str | None, int]]:
    # Walk PART and Item headers in document order. Returns triples of
    # (item_token, part_label_or_None, start_offset).
    # 10-K dedup is by item_token alone (item numbers are globally unique).
    # 10-Q dedup is by (part, item_token) because the same number appears in
    # both PART I and PART II with different meanings.
    # In both cases the LAST occurrence per key wins, which discards the
    # table-of-contents header in favor of the real section anchor.
    events: list[tuple[int, str, str]] = []
    for m in _PART_HEADER.finditer(text):
        roman = m.group(1).upper()
        arabic = _ROMAN_TO_ARABIC.get(roman)
        if arabic:
            events.append((m.start(), "part", arabic))
    for m in _ITEM_HEADER.finditer(text):
        token = m.group(1).upper()
        if form == "10-K" and token not in ITEM_TO_CANONICAL:
            continue
        events.append((m.start(), "item", token))
    events.sort(key=lambda e: e[0])

    current_part: str | None = None
    if form == "10-Q":
        last_qq: dict[tuple[str | None, str], int] = {}
        for offset, kind, value in events:
            if kind == "part":
                current_part = value
            else:
                last_qq[(current_part, value)] = offset
        return sorted(
            [(item, part, off) for (part, item), off in last_qq.items()],
            key=lambda x: x[2],
        )

    last_k: dict[str, tuple[str | None, int]] = {}
    for offset, kind, value in events:
        if kind == "part":
            current_part = value
        else:
            last_k[value] = (current_part, offset)
    return sorted(
        [(item, part, off) for item, (part, off) in last_k.items()],
        key=lambda x: x[2],
    )


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
