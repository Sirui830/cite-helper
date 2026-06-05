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
FINITE_VERB_RE = re.compile(
    r"\b("
    r"is|are|was|were|be|being|been|"
    r"has|have|had|do|does|did|"
    r"show|shows|showed|shown|find|finds|found|"
    r"suggest|suggests|suggested|indicate|indicates|indicated|"
    r"argue|argues|argued|examine|examines|examined|"
    r"investigate|investigates|investigated|use|uses|used|"
    r"reveal|reveals|revealed|demonstrate|demonstrates|demonstrated|"
    r"employ|employs|employed|prefer|prefers|preferred|"
    r"vary|varies|varied|affect|affects|affected"
    r")\b",
    re.IGNORECASE,
)


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
    if re.search(r"\bdoi\b|https?://|www\.|journal of|vol(?:ume)?\.", sentence, re.I):
        return True
    if re.search(r"\b(19|20)\d{2}[a-z]?\b", sentence) and re.search(
        r"\b(pp?\.|pages?|press|publisher|routledge|elsevier|springer|wiley)\b",
        sentence,
        re.I,
    ):
        return True
    return False


def _looks_like_title_sentence(sentence: str, section: str = "") -> bool:
    """Catch standalone article/book titles that are not usable evidence."""
    s = re.sub(r"\s+", " ", sentence).strip()
    if not s:
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", s)
    if not (4 <= len(words) <= 22):
        return False

    section_norm = section.lower().strip(" :.")
    likely_title_section = section_norm in {"header", "title"} or section_norm.startswith(
        "chunk_"
    )
    has_colon = ":" in s
    has_finite_verb = FINITE_VERB_RE.search(s) is not None
    starts_like_title = re.match(
        r"^(a|an|the|towards?|toward|responding|request|requests|"
        r"politeness|impoliteness|disagreeing|compliment|compliments|"
        r"computer-mediated|cross-cultural|intercultural|self-praise|"
        r"modeling|modelling|studying|understanding|exploring|teaching|"
        r"learning|making|communicating)\b",
        s,
        re.I,
    )
    title_case_words = sum(1 for w in words if w[:1].isupper())
    title_case_ratio = title_case_words / max(1, len(words))

    if likely_title_section and not has_finite_verb and (has_colon or starts_like_title):
        return True
    if has_colon and not has_finite_verb and title_case_ratio >= 0.35:
        return True
    if starts_like_title and not has_finite_verb and s.endswith("."):
        return True
    return False


def is_retrieval_noise(sentence: str, section: str = "") -> bool:
    """Return True for sentences that should not appear as citation hits."""
    if _looks_like_reference_entry(sentence):
        return True
    if _looks_like_title_sentence(sentence, section=section):
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
            if is_retrieval_noise(s, section=section.section):
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
