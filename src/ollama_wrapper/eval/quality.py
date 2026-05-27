from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalQualityReport:
    query_term_hit_rate: float
    answer_groundedness_score: float
    context_coverage_score: float
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QualityRegressionGate:
    min_query_term_hit_rate: float = 0.40
    min_answer_groundedness_score: float = 0.50
    min_context_coverage_score: float = 0.05


@dataclass(slots=True)
class QualityRegressionResult:
    sample_count: int
    avg_query_term_hit_rate: float
    avg_answer_groundedness_score: float
    avg_context_coverage_score: float
    passed: bool
    failures: list[str] = field(default_factory=list)


class RetrievalQualityEvaluator:
    """Lightweight lexical quality checks for retrieval and grounded responses."""

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t.strip(".,!?;:\"'()[]{}") for t in (text or "").lower().split() if t.strip()}

    def evaluate(self, query: str, contexts: list[str], answer: str) -> RetrievalQualityReport:
        query_terms = self._tokenize(query)
        answer_terms = self._tokenize(answer)
        context_terms = set()
        for ctx in contexts or []:
            context_terms.update(self._tokenize(ctx))

        if not query_terms:
            query_hit_rate = 0.0
        else:
            query_hit_rate = float(len(query_terms & context_terms) / len(query_terms))

        if not answer_terms:
            groundedness = 0.0
        else:
            groundedness = float(len(answer_terms & context_terms) / len(answer_terms))

        if not context_terms:
            coverage = 0.0
        else:
            coverage = float(len(context_terms & query_terms) / len(context_terms))

        warnings: list[str] = []
        if query_hit_rate < 0.4:
            warnings.append("low-query-context-overlap")
        if groundedness < 0.5:
            warnings.append("low-answer-groundedness")

        return RetrievalQualityReport(
            query_term_hit_rate=query_hit_rate,
            answer_groundedness_score=groundedness,
            context_coverage_score=coverage,
            warnings=warnings,
        )

    def evaluate_dataset(
        self,
        samples: list[dict[str, Any]],
        gate: QualityRegressionGate | None = None,
    ) -> QualityRegressionResult:
        if not samples:
            return QualityRegressionResult(
                sample_count=0,
                avg_query_term_hit_rate=0.0,
                avg_answer_groundedness_score=0.0,
                avg_context_coverage_score=0.0,
                passed=False,
                failures=["empty-dataset"],
            )

        effective_gate = gate or QualityRegressionGate()
        reports = [
            self.evaluate(
                query=str(sample.get("query", "")),
                contexts=list(sample.get("contexts", []) or []),
                answer=str(sample.get("answer", "")),
            )
            for sample in samples
        ]
        count = len(reports)
        avg_query = float(sum(r.query_term_hit_rate for r in reports) / count)
        avg_grounded = float(sum(r.answer_groundedness_score for r in reports) / count)
        avg_coverage = float(sum(r.context_coverage_score for r in reports) / count)

        failures: list[str] = []
        if avg_query < effective_gate.min_query_term_hit_rate:
            failures.append("query-term-hit-rate-below-threshold")
        if avg_grounded < effective_gate.min_answer_groundedness_score:
            failures.append("answer-groundedness-below-threshold")
        if avg_coverage < effective_gate.min_context_coverage_score:
            failures.append("context-coverage-below-threshold")

        return QualityRegressionResult(
            sample_count=count,
            avg_query_term_hit_rate=avg_query,
            avg_answer_groundedness_score=avg_grounded,
            avg_context_coverage_score=avg_coverage,
            passed=not failures,
            failures=failures,
        )
