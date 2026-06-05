"""Data schemas for cite-helper indexes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


INDEX_VERSION = 1
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


@dataclass
class IndexedSentence:
    """One sentence in the corpus, ready for retrieval."""

    sent_id: str
    paper_id: str
    citation_key: str
    pdf_path: str
    source_chunk_id: str
    section: str
    is_noise_section: bool
    sentence: str
    char_start: int
    char_end: int
    context_before: str
    context_after: str
    source_text_sha1: str
    model_name: str = DEFAULT_MODEL_NAME
    index_version: int = INDEX_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IndexMeta:
    """Top-level metadata for an index directory."""

    index_version: int
    model_name: str
    embedding_dim: int
    n_papers: int
    n_sentences: int
    pdf_hashes: dict = field(default_factory=dict)
    built_at: str = ""
    build_source: str = "pdf_folder"  # or "litreview_source_index"

    def to_dict(self) -> dict:
        return asdict(self)
