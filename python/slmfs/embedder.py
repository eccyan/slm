"""Embedding model interface."""

from abc import ABC, abstractmethod
import numpy as np


class Embedder(ABC):
    """Abstract embedding model interface."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text. Returns float32 array of shape (dim,)."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns float32 array of shape (N, dim)."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...


class MiniLMEmbedder(Embedder):
    """Default embedder using sentence-transformers/all-MiniLM-L6-v2 (384-dim)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._dim = 384

    def embed(self, text: str) -> np.ndarray:
        return (
            self._model.encode(text, normalize_embeddings=True)
            .astype(np.float32)
        )

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return (
            self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=True,
            )
            .astype(np.float32)
        )

    @property
    def dim(self) -> int:
        return self._dim
