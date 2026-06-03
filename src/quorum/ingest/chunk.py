from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase  # only for type hints

    from quorum.ingest.parse import Section

# Target sizes (Phase 3c). XLM-RoBERTa (BGE-M3's tokenizer) typically packs about
# 0.75 tokens per word; 750 tokens is roughly a page of dense prose.
DEFAULT_TARGET_TOKENS = 750
DEFAULT_OVERLAP_TOKENS = 100
BGE_M3_TOKENIZER_NAME = "BAAI/bge-m3"


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    section: str
    ordinal: int
    char_start: int  # absolute offset in the parsed filing text
    char_end: int

    def chunk_id(self, accession: str) -> str:
        return f"{accession}:{self.section}:{self.ordinal:04d}"


class Chunker:
    def __init__(
        self,
        *,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self._tok = tokenizer  # lazy load by default

    def _tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(BGE_M3_TOKENIZER_NAME)
        return self._tok

    def chunk_section(
        self,
        text: str,
        *,
        section: str,
        section_char_start: int = 0,
    ) -> list[Chunk]:
        if not text:
            return []
        tok = self._tokenizer()
        enc = tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if not ids:
            return []

        step = self.target_tokens - self.overlap_tokens
        out: list[Chunk] = []
        i = 0
        ordinal = 0
        while i < len(ids):
            j = min(i + self.target_tokens, len(ids))
            char_start_local, _ = offsets[i]
            _, char_end_local = offsets[j - 1]
            out.append(
                Chunk(
                    text=text[char_start_local:char_end_local],
                    section=section,
                    ordinal=ordinal,
                    char_start=section_char_start + char_start_local,
                    char_end=section_char_start + char_end_local,
                )
            )
            ordinal += 1
            if j >= len(ids):
                break
            i += step
        return out


def chunk_filing(
    sections: list[Section],
    *,
    exclude: set[str] | None = None,
    chunker: Chunker | None = None,
) -> list[Chunk]:
    from quorum.ingest.parse import SECTIONS_EXCLUDED_FROM_VECTOR_INDEX

    chunker = chunker or Chunker()
    skip = set(SECTIONS_EXCLUDED_FROM_VECTOR_INDEX) | (exclude or set())
    out: list[Chunk] = []
    for s in sections:
        if s.name in skip:
            continue
        out.extend(chunker.chunk_section(s.text, section=s.name, section_char_start=s.char_start))
    return out
