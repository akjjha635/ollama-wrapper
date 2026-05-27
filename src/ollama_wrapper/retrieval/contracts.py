from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Retriever(ABC):
    """Retrieval contract for hybrid candidate generation and reranking."""

    @abstractmethod
    def retrieve_and_rerank(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        sync_mode: bool = True,
    ) -> str:
        raise NotImplementedError


class Reranker(ABC):
    """Reranker contract for choosing a best candidate from retrieved chunks."""

    @abstractmethod
    def choose(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        wrapper: Any,
    ) -> int:
        raise NotImplementedError


class DenseBackend(ABC):
    """Dense score backend contract for semantic candidate scoring."""

    @abstractmethod
    def score(
        self,
        matrix: np.ndarray,
        query_vector: np.ndarray,
        warn: Any,
    ) -> np.ndarray:
        raise NotImplementedError
