from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RAGCandidate:
    text: str
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizationConfig:
    semantic_weight: float = 0.7
    lexical_weight: float = 0.3
    token_budget: int = 1200
    max_context_items: int = 8
    query_type: str = "general"
    diversity_lambda: float = 0.15


@dataclass(slots=True)
class ContextOptimizationResult:
    selected_context: list[str] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    combined_scores: list[float] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    token_estimate: int = 0
    token_budget: int = 0
    dropped_candidates: int = 0


@dataclass(slots=True)
class OptimizationSummary:
    candidate_count: int = 0
    selected_count: int = 0
    token_estimate: int = 0
    token_budget: int = 0
    score_weights: dict[str, float] = field(default_factory=dict)
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
