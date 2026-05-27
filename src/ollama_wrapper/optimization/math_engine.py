from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import (
    ContextOptimizationResult,
    OptimizationConfig,
    OptimizationSummary,
    RAGCandidate,
)


class MathematicalOptimizationLayer:
    """Numerical optimization layer for context selection and RAG scoring."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        words = len(text.split())
        return max(1, int(math.ceil(words * 1.3)))

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        v_min = float(values.min())
        v_max = float(values.max())
        if v_max <= v_min:
            return np.ones_like(values)
        return (values - v_min) / (v_max - v_min + 1e-9)

    @staticmethod
    def _calibration_factors(query_type: str) -> tuple[float, float]:
        q = (query_type or "general").lower()
        if q in {"factoid", "lookup", "factual"}:
            return 0.60, 1.80
        if q in {"reasoning", "analysis"}:
            return 1.80, 0.60
        if q in {"code", "technical"}:
            return 1.05, 1.05
        return 1.0, 1.0

    @staticmethod
    def _candidate_source(candidate: RAGCandidate) -> str:
        metadata = candidate.metadata or {}
        return str(
            metadata.get("source")
            or metadata.get("doc_id")
            or metadata.get("path")
            or metadata.get("title")
            or ""
        )

    def _diversity_penalty(
        self,
        candidate: RAGCandidate,
        selected_candidates: list[RAGCandidate],
        diversity_lambda: float,
    ) -> float:
        if not selected_candidates or diversity_lambda <= 0:
            return 0.0
        source = self._candidate_source(candidate)
        if not source:
            return 0.0
        if any(self._candidate_source(existing) == source for existing in selected_candidates):
            return float(diversity_lambda)
        return 0.0

    @staticmethod
    def _confidence(scores: list[float]) -> list[float]:
        if not scores:
            return []
        arr = np.array(scores, dtype=np.float64)
        baseline = float(np.max(arr))
        exp = np.exp(arr - baseline)
        total = float(np.sum(exp))
        if total <= 0.0:
            return [0.0 for _ in scores]
        return [float(v / total) for v in exp]

    def optimize_context(
        self,
        candidates: list[RAGCandidate],
        config: OptimizationConfig,
    ) -> ContextOptimizationResult:
        if not candidates:
            return ContextOptimizationResult(token_budget=config.token_budget)

        semantic = np.array([c.semantic_score for c in candidates], dtype=np.float32)
        lexical = np.array([c.lexical_score for c in candidates], dtype=np.float32)

        norm_semantic = self._normalize(semantic)
        norm_lexical = self._normalize(lexical)

        sem_factor, lex_factor = self._calibration_factors(config.query_type)
        weighted_semantic = norm_semantic * sem_factor
        weighted_lexical = norm_lexical * lex_factor

        combined = (config.semantic_weight * weighted_semantic) + (config.lexical_weight * weighted_lexical)
        remaining = set(range(len(candidates)))

        selected_text: list[str] = []
        selected_indices: list[int] = []
        selected_scores: list[float] = []
        selected_candidates: list[RAGCandidate] = []
        token_total = 0

        while remaining:
            if len(selected_indices) >= max(1, config.max_context_items):
                break

            best_idx: int | None = None
            best_score = float("-inf")
            unseen_sources = [
                idx
                for idx in remaining
                if self._candidate_source(candidates[idx])
                and not any(
                    self._candidate_source(existing) == self._candidate_source(candidates[idx])
                    for existing in selected_candidates
                )
            ]
            candidate_pool = unseen_sources if unseen_sources else list(remaining)

            for idx in candidate_pool:
                candidate = candidates[idx]
                penalty = self._diversity_penalty(candidate, selected_candidates, config.diversity_lambda)
                score = float(combined[idx]) - penalty
                if score > best_score:
                    best_score = score
                    best_idx = int(idx)

            if best_idx is None:
                break

            idx = best_idx
            remaining.remove(idx)
            candidate_tokens = self.estimate_tokens(candidates[idx].text)
            if token_total + candidate_tokens > max(1, config.token_budget):
                continue
            selected_text.append(candidates[idx].text)
            selected_indices.append(int(idx))
            selected_scores.append(best_score)
            selected_candidates.append(candidates[idx])
            token_total += candidate_tokens

        confidence_scores = self._confidence(selected_scores)

        return ContextOptimizationResult(
            selected_context=selected_text,
            selected_indices=selected_indices,
            combined_scores=selected_scores,
            confidence_scores=confidence_scores,
            token_estimate=token_total,
            token_budget=config.token_budget,
            dropped_candidates=max(0, len(candidates) - len(selected_indices)),
        )

    def summarize(
        self,
        candidates: list[RAGCandidate],
        result: ContextOptimizationResult,
        config: OptimizationConfig,
    ) -> OptimizationSummary:
        return OptimizationSummary(
            candidate_count=len(candidates),
            selected_count=len(result.selected_indices),
            token_estimate=result.token_estimate,
            token_budget=result.token_budget,
            score_weights={
                "semantic": config.semantic_weight,
                "lexical": config.lexical_weight,
            },
            avg_confidence=(
                float(sum(result.confidence_scores) / len(result.confidence_scores))
                if result.confidence_scores
                else 0.0
            ),
            max_confidence=max(result.confidence_scores) if result.confidence_scores else 0.0,
        )

    @staticmethod
    def compute_bm25_score(
        query_terms: list[str],
        doc_terms: list[str],
        doc_len: int,
        avg_doc_len: float,
        term_df: dict[str, int],
        total_docs: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        if not query_terms or not doc_terms or total_docs <= 0:
            return 0.0

        score = 0.0
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf == 0:
                continue
            df_t = term_df.get(term, 0)
            idf = math.log((total_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / (avg_doc_len + 1e-9)))
            score += idf * (numerator / denominator)
        return float(score)

    @staticmethod
    def parse_candidates(raw_candidates: list[dict[str, Any]] | None) -> list[RAGCandidate]:
        if not raw_candidates:
            return []
        parsed: list[RAGCandidate] = []
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            parsed.append(
                RAGCandidate(
                    text=text,
                    semantic_score=float(item.get("semantic_score", 0.0)),
                    lexical_score=float(item.get("lexical_score", 0.0)),
                    metadata=item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {},
                )
            )
        return parsed
