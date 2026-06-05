"""Sentence-level chunking with pysbd, dedup, and context windows."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from .pdf_parser import PdfSection
from .schemas import IndexedSentence, DEFAULT_MODEL_NAME, INDEX_VERSION


MIN_SENT_LEN = 25
MAX_SENT_LEN = 500
CONTEXT_WINDOW = 1  # ±N sentences shown around each hit


def _normalize_for_dedup(text: str) -> str:
    """Aggressive normalization used only for dedup hashing."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _looks_like_reference_entry(sentence: str) -> bool:
    """Heuristic: reference list entry that survived NOISE section filter.

    These often appear inside Abstract / chunk_X when PDF parsing pulls
    them in. Catch obvious year-citation patterns like
    'Smith, J., 2020. Title. Journal X, 12-34.'
    """
    if re.search(r"\b(19|20)\d{2}[a-z]?\b.*\.\s*[A-Z]", sentence) and (
        sentence.count(",") >= 3 or re.search(r"\bpp?\.\s*\d", sentence)
    ):
        return True
    return False


@dataclass
class _ChunkerState:
    paper_id: str
    seen_normalized: set[str]


def chunk_sections(
    sections: list[PdfSection],
    citation_key: str,
    seg=None,
) -> list[IndexedSentence]:
    """Split sections into sentences, attach context, dedup within paper."""
    if seg is None:
        import pysbd
        seg = pysbd.Segmenter(language="en", clean=False)

    state = _ChunkerState(
        paper_id=sections[0].paper_id if sections else "",
        seen_normalized=set(),
    )
    out: list[IndexedSentence] = []

    for section in sections:
        if not section.text:
            continue
        raw_sentences = [s.strip() for s in seg.segment(section.text) if s.strip()]
        # Filter by length and reference-entry heuristic
        filtered_pairs: list[tuple[int, str]] = []
        for i, s in enumerate(raw_sentences):
            if not (MIN_SENT_LEN <= len(s) <= MAX_SENT_LEN):
                continue
            if _looks_like_reference_entry(s):
                continue
            filtered_pairs.append((i, s))

        for orig_idx, sentence in filtered_pairs:
            norm = _normalize_for_dedup(sentence)
            if not norm or norm in state.seen_normalized:
                continue
            state.seen_normalized.add(norm)

            ctx_before_parts = [
                raw_sentences[j]
                for j in range(max(0, orig_idx - CONTEXT_WINDOW), orig_idx)
            ]
            ctx_after_parts = [
                raw_sentences[j]
                for j in range(
                    orig_idx + 1,
                    min(len(raw_sentences), orig_idx + 1 + CONTEXT_WINDOW),
                )
            ]

            # Best-effort char offset in section.text (find on cleaned)
            try:
                start_idx = section.text.index(sentence)
                end_idx = start_idx + len(sentence)
            except ValueError:
                start_idx, end_idx = -1, -1

            sent_id = "S_" + hashlib.sha1(
                f"{section.sha1}|{orig_idx}|{sentence[:120]}".encode("utf-8")
            ).hexdigest()[:12]

            out.append(
                IndexedSentence(
                    sent_id=sent_id,
                    paper_id=state.paper_id,
                    citation_key=citation_key,
                    pdf_path=section.pdf_path,
                    source_chunk_id=section.sha1,
                    section=section.section,
                    is_noise_section=section.is_noise,
                    sentence=sentence,
                    char_start=start_idx,
                    char_end=end_idx,
                    context_before=" ".join(ctx_before_parts),
                    context_after=" ".join(ctx_after_parts),
                    source_text_sha1=section.sha1,
                    model_name=DEFAULT_MODEL_NAME,
                    index_version=INDEX_VERSION,
                )
            )

    return out


def filter_noise(sentences: list[IndexedSentence]) -> list[IndexedSentence]:
    """Drop sentences from NOISE_SECTIONS (References, Acks, etc)."""
    return [s for s in sentences if not s.is_noise_section]
