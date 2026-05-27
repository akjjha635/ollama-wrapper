from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import DenseBackend


class LinearDenseBackend(DenseBackend):
    def score(self, matrix: np.ndarray, query_vector: np.ndarray, warn: Any) -> np.ndarray:
        _ = warn
        return np.dot(matrix, query_vector)


class FaissDenseBackend(DenseBackend):
    """FAISS-backed dense scoring with automatic linear fallback."""

    def __init__(self) -> None:
        self._warned = False

    def score(self, matrix: np.ndarray, query_vector: np.ndarray, warn: Any) -> np.ndarray:
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(np.asarray(matrix, dtype=np.float32))
            scores, _ = index.search(np.asarray([query_vector], dtype=np.float32), matrix.shape[0])
            return scores[0]
        except Exception:
            if not self._warned and callable(warn):
                warn("FAISS backend unavailable. Falling back to linear dense scoring.")
                self._warned = True
            return np.dot(matrix, query_vector)
