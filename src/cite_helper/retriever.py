"""Load a built index and run top-k cosine retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunker import is_retrieval_noise
from .embedder import Embedder
from .indexer import INDEX_DIRNAME, index_dir
from .schemas import IndexedSentence


@dataclass
class SearchHit:
    rank: int
    score: float
    sentence: IndexedSentence
    context_before: str | None = None
    context_after: str | None = None

    def to_dict(self) -> dict:
        data = {
            "rank": self.rank,
            "score": self.score,
            **self.sentence.to_dict(),
        }
        if self.context_before is not None:
            data["context_before"] = self.context_before
        if self.context_after is not None:
            data["context_after"] = self.context_after
        return data


class Index:
    def __init__(self, folder: Path):
        self.folder = folder.resolve()
        self.dir = index_dir(self.folder)
        if not self.dir.exists():
            raise RuntimeError(
                f"No index found at {self.dir}. "
                f"Run: cite-helper build {folder}"
            )
        with (self.dir / "meta.json").open() as f:
            self.meta = json.load(f)
        with (self.dir / "sentences.jsonl").open() as f:
            self.sentences = [
                IndexedSentence(**json.loads(line))
                for line in f
                if line.strip()
            ]
        self.embeddings = np.load(self.dir / "sentence_embeddings.npy")
        if self.embeddings.shape[0] != len(self.sentences):
            raise RuntimeError(
                "Index corrupt: embedding count != sentence count "
                f"({self.embeddings.shape[0]} vs {len(self.sentences)})"
            )
        self._embedder: Embedder | None = None
        self._chunk_sentence_indices: dict[str, list[int]] = {}
        for i, sentence in enumerate(self.sentences):
            self._chunk_sentence_indices.setdefault(
                sentence.source_chunk_id, []
            ).append(i)
        for indices in self._chunk_sentence_indices.values():
            indices.sort(
                key=lambda i: (
                    self.sentences[i].char_start < 0,
                    self.sentences[i].char_start
                    if self.sentences[i].char_start >= 0
                    else i,
                )
            )

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(model_name=self.meta["model_name"])
        return self._embedder

    @staticmethod
    def _matches_text_filter(value: str, needle: str | None) -> bool:
        if not needle:
            return True
        return needle.lower() in value.lower()

    def _candidate_indices(
        self,
        paper: str | None = None,
        section: str | None = None,
        include_noise: bool = False,
    ) -> list[int]:
        indices: list[int] = []
        for i, sentence in enumerate(self.sentences):
            if paper:
                paper_haystack = " ".join(
                    [
                        sentence.paper_id,
                        sentence.citation_key,
                        Path(sentence.pdf_path).name,
                    ]
                )
                if not self._matches_text_filter(paper_haystack, paper):
                    continue
            if section and not self._matches_text_filter(sentence.section, section):
                continue
            if not include_noise and is_retrieval_noise(
                sentence.sentence, section=sentence.section
            ):
                continue
            indices.append(i)
        return indices

    def _context_for_index(self, index: int, context_window: int) -> tuple[str, str]:
        if context_window <= 0:
            return "", ""
        sentence = self.sentences[index]
        chunk_indices = self._chunk_sentence_indices.get(sentence.source_chunk_id, [])
        try:
            pos = chunk_indices.index(index)
        except ValueError:
            return sentence.context_before, sentence.context_after
        before_indices = chunk_indices[max(0, pos - context_window):pos]
        after_indices = chunk_indices[pos + 1:pos + 1 + context_window]
        before = " ".join(self.sentences[i].sentence for i in before_indices)
        after = " ".join(self.sentences[i].sentence for i in after_indices)
        return before, after

    def find(
        self,
        query: str,
        k: int = 5,
        paper: str | None = None,
        section: str | None = None,
        context_window: int = 1,
        include_noise: bool = False,
    ) -> list[SearchHit]:
        if k <= 0:
            return []
        candidate_indices = self._candidate_indices(
            paper=paper,
            section=section,
            include_noise=include_noise,
        )
        if not candidate_indices:
            return []
        q_emb = self.embedder.embed_query(query)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sims = self.embeddings @ q_emb
        sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
        candidate_sims = sims[candidate_indices]
        if k >= len(candidate_sims):
            selected_pos = np.argsort(-candidate_sims)
        else:
            selected_pos = np.argpartition(-candidate_sims, k - 1)[:k]
            selected_pos = selected_pos[np.argsort(-candidate_sims[selected_pos])]
        hits: list[SearchHit] = []
        for rank, pos in enumerate(selected_pos, start=1):
            i = candidate_indices[int(pos)]
            before, after = self._context_for_index(i, context_window)
            hits.append(
                SearchHit(
                    rank=rank,
                    score=float(sims[i]),
                    sentence=self.sentences[i],
                    context_before=before,
                    context_after=after,
                )
            )
        return hits

    def verify(self, text: str) -> list[SearchHit]:
        """Find sentences whose normalized form contains the given text.

        Whitespace-tolerant substring search. Used to confirm a quote
        appears verbatim somewhere in the corpus.
        """
        import re
        norm_q = re.sub(r"\s+", "", text).lower()
        if not norm_q:
            return []
        hits: list[SearchHit] = []
        for i, s in enumerate(self.sentences):
            norm_s = re.sub(r"\s+", "", s.sentence).lower()
            if norm_q in norm_s:
                hits.append(
                    SearchHit(
                        rank=len(hits) + 1,
                        score=1.0,
                        sentence=s,
                    )
                )
                if len(hits) >= 10:
                    break
        return hits
