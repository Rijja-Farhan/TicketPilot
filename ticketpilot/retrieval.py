"""Chunk retrieval over a built index: BM25, dense (cosine), or hybrid (reciprocal rank fusion).

Status filtering happens here, before ranking: default is current chunks only; history questions add superseded.
Drafts were never in force and are returned only when asked for explicitly (the evals do, to measure stale hits).
"""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ticketpilot.embeddings import Embedder, get_embedder
from ticketpilot.ingest.chunk import Chunk

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it long many much my of on or our the "
    "their there this to was we what when which who will with you your".split())
RRF_K = 60
CURRENT = ("current",)
WITH_HISTORY = ("current", "superseded")
ALL_STATUSES = ("current", "superseded", "draft")


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        self.chunks, self.vectors, self.embedder = chunks, vectors, embedder
        self._bm25 = BM25Okapi([tokens(c.text) for c in chunks]) if chunks else None

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder | None = None) -> "Index":
        embedder = embedder or get_embedder()
        return cls(chunks, embedder.embed_documents([c.text for c in chunks]), embedder)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "chunks.jsonl").open("w") as f:
            f.writelines(json.dumps(asdict(c)) + "\n" for c in self.chunks)
        np.save(path / "vectors.npy", self.vectors)
        (path / "meta.json").write_text(json.dumps({"embedder": self.embedder.name, "chunks": len(self.chunks)}))

    @classmethod
    def load(cls, path: Path) -> "Index":
        meta = json.loads((path / "meta.json").read_text())
        chunks = [Chunk(**json.loads(line)) for line in (path / "chunks.jsonl").read_text().splitlines()]
        return cls(chunks, np.load(path / "vectors.npy"), get_embedder(meta["embedder"]))

    def _allowed(self, statuses: tuple[str, ...]) -> list[int]:
        return [i for i, c in enumerate(self.chunks) if c.status in statuses]

    def _bm25_ranking(self, query: str, allowed: list[int]) -> list[tuple[int, float]]:
        q = tokens(query)
        if not q or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(((i, float(scores[i])) for i in allowed if scores[i] > 0), key=lambda p: (-p[1], p[0]))
        return ranked

    def _dense_ranking(self, query: str, allowed: list[int]) -> list[tuple[int, float]]:
        if not allowed:
            return []
        q = self.embedder.embed_queries([query])[0]
        sims = self.vectors[allowed] @ q
        return sorted(((allowed[j], float(s)) for j, s in enumerate(sims)), key=lambda p: (-p[1], p[0]))

    def search(self, query: str, k: int = 3, mode: str = "hybrid", statuses: tuple[str, ...] = CURRENT,
               require_lexical: bool = False) -> list[Hit]:
        """require_lexical: return nothing unless some query word occurs in an allowed chunk. Dense retrieval always
        ranks something, so this keeps "no match -> no answer attempt" for questions unrelated to the KB."""
        allowed = self._allowed(statuses)
        if require_lexical and not self._bm25_ranking(query, allowed):
            return []
        if mode == "bm25":
            ranking = self._bm25_ranking(query, allowed)
        elif mode == "dense":
            ranking = self._dense_ranking(query, allowed)
        elif mode == "hybrid":
            fused: dict[int, float] = {}
            for ranked in (self._bm25_ranking(query, allowed), self._dense_ranking(query, allowed)):
                for rank, (i, _) in enumerate(ranked, 1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
            ranking = sorted(fused.items(), key=lambda p: (-p[1], p[0]))
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        return [Hit(self.chunks[i], s) for i, s in ranking[:k]]
