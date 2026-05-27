#!/usr/bin/env python3
"""Run retrieval quality regression gates against a JSON sample dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from ollama_wrapper.eval import QualityRegressionGate, RetrievalQualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval quality regression checks.")
    parser.add_argument("--dataset", required=True, help="Path to JSON dataset: [{query, contexts, answer}, ...]")
    parser.add_argument("--min-query-hit", type=float, default=0.40)
    parser.add_argument("--min-groundedness", type=float, default=0.50)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.dataset, "r", encoding="utf-8") as handle:
        samples = json.load(handle)

    evaluator = RetrievalQualityEvaluator()
    gate = QualityRegressionGate(
        min_query_term_hit_rate=args.min_query_hit,
        min_answer_groundedness_score=args.min_groundedness,
        min_context_coverage_score=args.min_coverage,
    )
    result = evaluator.evaluate_dataset(samples=samples, gate=gate)

    payload = {
        "sample_count": result.sample_count,
        "avg_query_term_hit_rate": result.avg_query_term_hit_rate,
        "avg_answer_groundedness_score": result.avg_answer_groundedness_score,
        "avg_context_coverage_score": result.avg_context_coverage_score,
        "passed": result.passed,
        "failures": result.failures,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
