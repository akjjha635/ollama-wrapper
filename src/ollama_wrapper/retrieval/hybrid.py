from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from ollama_wrapper.optimization import OptimizationConfig, RAGCandidate

from .contracts import DenseBackend, Reranker, Retriever
from .registry import DenseBackendRegistry, create_default_dense_backend_registry
from .rerankers import LLMChunkIDReranker


class HybridRetriever(Retriever):
    """Hybrid dense + BM25 retriever with LLM rerank over selected candidates."""

    def __init__(
        self,
        wrapper: Any,
        backend: str | DenseBackend = "linear",
        reranker: Reranker | None = None,
        backend_registry: DenseBackendRegistry | None = None,
    ) -> None:
        self.wrapper = wrapper
        self.backend_registry = backend_registry or create_default_dense_backend_registry()
        self.backend = self._resolve_backend(backend)
        self.reranker = reranker or LLMChunkIDReranker()

    def _resolve_backend(self, backend: str | DenseBackend) -> DenseBackend:
        if isinstance(backend, DenseBackend):
            return backend
        backend_name = str(backend or "linear").lower()
        try:
            return self.backend_registry.create(backend_name)
        except KeyError:
            self.wrapper_logger_warning(f"Unknown retrieval backend '{backend_name}'. Falling back to linear.")
            return self.backend_registry.create("linear")

    def _compute_bm25_score(self, query: str, doc_idx: int) -> float:
        query_terms = query.lower().split()
        doc_terms = self.wrapper.vector_database[doc_idx]["text"].lower().split()
        doc_len = self.wrapper.doc_lens[doc_idx]

        return self.wrapper._optimization_layer.compute_bm25_score(
            query_terms=query_terms,
            doc_terms=doc_terms,
            doc_len=doc_len,
            avg_doc_len=float(self.wrapper.avg_doc_len),
            term_df=self.wrapper.df,
            total_docs=len(self.wrapper.vector_database),
            k1=self.wrapper.k1,
            b=self.wrapper.b,
        )

    def retrieve_and_rerank(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        sync_mode: bool = True,
    ) -> str:
        if not self.wrapper.vector_database:
            return ""

        pool_indices = list(range(len(self.wrapper.vector_database)))
        if metadata_filter:
            pool_indices = [
                i
                for i in pool_indices
                if all(self.wrapper.vector_database[i]["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        if not pool_indices:
            return ""

        candidates: list[dict[str, Any]] = []
        try:
            if sync_mode:
                res = self.wrapper._sync_client.embed(model=self.wrapper.embed_model, input=query)
            else:
                loop = asyncio.get_running_loop()
                coro = self.wrapper._async_client.embed(model=self.wrapper.embed_model, input=query)
                res = asyncio.run_coroutine_threadsafe(coro, loop).result()

            q_vec = np.array(res.embeddings[0], dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec /= q_norm

            matrix = np.array([self.wrapper.vector_database[i]["vector"] for i in pool_indices])
            vector_scores = self.backend.score(matrix, q_vec, self.wrapper_logger_warning)
            bm25_scores = np.array([self._compute_bm25_score(query, i) for i in pool_indices], dtype=np.float32)

            optimization_candidates = [
                RAGCandidate(
                    text=self.wrapper.vector_database[idx]["text"],
                    semantic_score=float(vector_scores[pos]),
                    lexical_score=float(bm25_scores[pos]),
                    metadata=self.wrapper.vector_database[idx].get("metadata", {}),
                )
                for pos, idx in enumerate(pool_indices)
            ]
            optimization_config = OptimizationConfig(
                semantic_weight=0.7,
                lexical_weight=0.3,
                token_budget=1000000,
                max_context_items=min(5, len(pool_indices)),
            )
            optimization_result = self.wrapper._optimization_layer.optimize_context(
                optimization_candidates,
                optimization_config,
            )

            candidate_positions = optimization_result.selected_indices or [0]
            candidates = [self.wrapper.vector_database[pool_indices[pos]] for pos in candidate_positions]

            chosen_id = self.reranker.choose(query=query, candidates=candidates, wrapper=self.wrapper)
            if 0 <= chosen_id < len(candidates):
                return candidates[chosen_id]["text"]
        except Exception as e:
            self.wrapper_logger_warning(f"Reranking failed: {e}. Falling back to best combined score.")

        return candidates[0]["text"] if candidates else self.wrapper.vector_database[pool_indices[0]]["text"]

    def wrapper_logger_warning(self, message: str) -> None:
        logger = getattr(self.wrapper, "logger", None)
        if logger is not None:
            logger.warning(message)
            return
        try:
            import logging

            logging.getLogger(__name__).warning(message)
        except Exception:
            pass
