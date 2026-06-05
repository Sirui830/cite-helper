"""Load a built index and run top-k cosine retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedder import Embedder
from .indexer import INDEX_DIRNAME, index_dir
from .schemas import IndexedSentence


@dataclass
class SearchHit:
    rank: int
    score: float
    sentence: IndexedSentence

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": self.score,
            **self.sentence.to_dict(),
        }


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

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(model_name=self.meta["model_name"])
        return self._embedder

    def find(self, query: str, k: int = 5) -> list[SearchHit]:
        q_emb = self.embedder.embed_query(query)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sims = self.embeddings @ q_emb
        sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
        if k >= len(sims):
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, k)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [
            SearchHit(
                rank=rank + 1,
                score=float(sims[i]),
                sentence=self.sentences[i],
            )
            for rank, i in enumerate(top_idx)
        ]

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
