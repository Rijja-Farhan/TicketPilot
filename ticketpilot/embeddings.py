"""One embedding interface. `fastembed` is a real local model (live runs); `hash` is a deterministic bag of hashed
words so the offline suite needs no download. Select with EMBEDDINGS_PROVIDER (default fastembed)."""
import hashlib
import os
import re
from typing import Protocol

import numpy as np

FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", t.lower())
            for tok in words + [f"{a} {b}" for a, b in zip(words, words[1:])]:
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise(out)

    embed_queries = embed_documents


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model: str = FASTEMBED_MODEL):
        from fastembed import TextEmbedding  # heavy import, only for live runs
        self._model = TextEmbedding(model)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.array(list(self._model.embed(texts))))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.array(list(self._model.query_embed(texts))))


_cache: dict[str, Embedder] = {}


def get_embedder(name: str | None = None) -> Embedder:
    name = name or os.environ.get("EMBEDDINGS_PROVIDER", "fastembed")
    if name not in _cache:
        if name == "hash":
            _cache[name] = HashEmbedder()
        elif name == "fastembed":
            _cache[name] = FastEmbedEmbedder()
        else:
            raise ValueError(f"Unknown EMBEDDINGS_PROVIDER {name!r}; expected 'fastembed' or 'hash'")
    return _cache[name]
