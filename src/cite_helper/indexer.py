"""Build .cite_helper_index/ from a PDF folder or a LitReview+ source_index.jsonl."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunker import chunk_sections, filter_noise
from .embedder import Embedder
from .pdf_parser import PdfSection, file_sha256, parse_pdf
from .schemas import (
    DEFAULT_MODEL_NAME,
    INDEX_VERSION,
    IndexMeta,
    IndexedSentence,
)


INDEX_DIRNAME = ".cite_helper_index"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(text: str, max_words: int = 5) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    stop = {"the", "a", "an", "of", "and", "in", "for", "with", "on", "to"}
    selected = [w for w in words if w.lower() not in stop][:max_words]
    if not selected:
        selected = words[:max_words] or ["paper"]
    return "_".join(selected)


def _citation_key_for(index: int, paper_id: str) -> str:
    return f"P{index:02d}_{_slugify(paper_id)}"


def index_dir(target: Path) -> Path:
    return target / INDEX_DIRNAME


def _say(msg: str) -> None:
    print(msg, flush=True)


def build_from_pdf_folder(folder: Path, model_name: str = DEFAULT_MODEL_NAME) -> dict:
    """Walk *.pdf files in folder, parse, chunk, embed, and persist."""
    folder = folder.resolve()
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {folder}")

    t_start = time.time()
    import pysbd
    seg = pysbd.Segmenter(language="en", clean=False)

    all_sentences: list[IndexedSentence] = []
    pdf_hashes: dict[str, str] = {}

    _say(f"[1/3] Parsing {len(pdfs)} PDFs...")
    for i, pdf_path in enumerate(pdfs, start=1):
        size_mb = pdf_path.stat().st_size / 1024 / 1024
        size_warn = " (LARGE — may take 1-2 min)" if size_mb > 5 else ""
        _say(f"  [{i}/{len(pdfs)}] parsing {pdf_path.name} ({size_mb:.1f}MB){size_warn}")
        try:
            sections = parse_pdf(pdf_path)
        except Exception as exc:
            _say(f"    FAILED: {exc}")
            continue
        if not sections:
            _say(f"    EMPTY")
            continue
        citation_key = _citation_key_for(i, sections[0].paper_id)
        sentences = chunk_sections(sections, citation_key=citation_key, seg=seg)
        sentences = filter_noise(sentences)
        all_sentences.extend(sentences)
        pdf_hashes[pdf_path.name] = file_sha256(pdf_path)
        _say(f"    -> {len(sentences)} sentences")

    if not all_sentences:
        raise RuntimeError("No sentences indexed — all PDFs failed or were empty.")

    _say(f"\n[2/3] Loading model {model_name}...")
    embedder = Embedder(model_name=model_name)

    _say(f"\n[3/3] Embedding {len(all_sentences)} sentences...")
    embeddings = embedder.embed_passages([s.sentence for s in all_sentences])

    out_dir = index_dir(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    sent_path = out_dir / "sentences.jsonl"
    with sent_path.open("w", encoding="utf-8") as f:
        for s in all_sentences:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    np.save(out_dir / "sentence_embeddings.npy", embeddings)

    meta = IndexMeta(
        index_version=INDEX_VERSION,
        model_name=model_name,
        embedding_dim=embedder.dim,
        n_papers=len(pdf_hashes),
        n_sentences=len(all_sentences),
        pdf_hashes=pdf_hashes,
        built_at=_now_iso(),
        build_source="pdf_folder",
    )
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)

    return {
        "folder": str(folder),
        "index_dir": str(out_dir),
        "n_papers": len(pdf_hashes),
        "n_sentences": len(all_sentences),
        "took_seconds": round(time.time() - t_start, 1),
        "index_size_mb": round(
            (sent_path.stat().st_size + (embeddings.nbytes)) / 1024 / 1024, 1
        ),
    }


def build_from_source_index(
    source_index_path: Path,
    output_folder: Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    """Build index from a pre-existing LitReview+ source_index.jsonl.

    Skips PDF parsing; uses the chunk text already extracted.
    """
    source_index_path = source_index_path.resolve()
    output_folder = output_folder.resolve()
    if not source_index_path.exists():
        raise RuntimeError(f"Not found: {source_index_path}")

    t_start = time.time()
    import pysbd
    seg = pysbd.Segmenter(language="en", clean=False)

    pseudo_sections: dict[str, list[PdfSection]] = {}
    citation_key_by_paper: dict[str, str] = {}
    with source_index_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            paper_id = row.get("paper_id", "?")
            citation_key_by_paper.setdefault(
                paper_id, row.get("citation_key", paper_id)
            )
            pseudo_sections.setdefault(paper_id, []).append(
                PdfSection(
                    paper_id=paper_id,
                    pdf_path=row.get("pdf_source", ""),
                    section=row.get("section", "Unknown"),
                    text=row.get("text", ""),
                    sha1=row.get("text_sha1", row.get("chunk_id", "")),
                    is_noise=False,
                )
            )

    all_sentences: list[IndexedSentence] = []
    for paper_id, sects in pseudo_sections.items():
        if not sects:
            continue
        citation_key = citation_key_by_paper.get(paper_id, paper_id)
        sentences = chunk_sections(sects, citation_key=citation_key, seg=seg)
        from .pdf_parser import is_noise_section
        for s in sentences:
            s.is_noise_section = is_noise_section(s.section)
        sentences = filter_noise(sentences)
        all_sentences.extend(sentences)
        _say(f"  {paper_id}: {len(sentences)} sentences")

    if not all_sentences:
        raise RuntimeError("No sentences indexed from source_index.")

    _say(f"\nLoading model {model_name}...")
    embedder = Embedder(model_name=model_name)

    _say(f"Embedding {len(all_sentences)} sentences...")
    embeddings = embedder.embed_passages([s.sentence for s in all_sentences])

    out_dir = index_dir(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "sentences.jsonl").open("w", encoding="utf-8") as f:
        for s in all_sentences:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    np.save(out_dir / "sentence_embeddings.npy", embeddings)

    meta = IndexMeta(
        index_version=INDEX_VERSION,
        model_name=model_name,
        embedding_dim=embedder.dim,
        n_papers=len(pseudo_sections),
        n_sentences=len(all_sentences),
        pdf_hashes={},
        built_at=_now_iso(),
        build_source="litreview_source_index",
    )
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)

    return {
        "source_index": str(source_index_path),
        "index_dir": str(out_dir),
        "n_papers": len(pseudo_sections),
        "n_sentences": len(all_sentences),
        "took_seconds": round(time.time() - t_start, 1),
    }
