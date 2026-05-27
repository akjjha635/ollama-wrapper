import unittest

from ollama_wrapper.eval import QualityRegressionGate, RetrievalQualityEvaluator


class TestEvalQuality(unittest.TestCase):
    def test_quality_evaluator_flags_low_overlap(self):
        evaluator = RetrievalQualityEvaluator()
        report = evaluator.evaluate(
            query="vector database indexing",
            contexts=["weather report", "sports scores"],
            answer="it depends on embeddings",
        )
        self.assertIn("low-query-context-overlap", report.warnings)
        self.assertLess(report.query_term_hit_rate, 0.4)

    def test_quality_evaluator_scores_grounded_answer(self):
        evaluator = RetrievalQualityEvaluator()
        report = evaluator.evaluate(
            query="token budget",
            contexts=["token budget protects context window", "budget mode can truncate prompts"],
            answer="token budget can truncate prompts",
        )
        self.assertGreaterEqual(report.answer_groundedness_score, 0.5)

    def test_quality_dataset_regression_gate(self):
        evaluator = RetrievalQualityEvaluator()
        samples = [
            {
                "query": "token budget",
                "contexts": ["token budget can truncate prompts"],
                "answer": "token budget can truncate prompts",
            },
            {
                "query": "rate limit",
                "contexts": ["rate limit controls qps burst"],
                "answer": "rate limit controls qps",
            },
        ]
        result = evaluator.evaluate_dataset(
            samples,
            gate=QualityRegressionGate(
                min_query_term_hit_rate=0.2,
                min_answer_groundedness_score=0.2,
                min_context_coverage_score=0.01,
            ),
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.sample_count, 2)


if __name__ == "__main__":
    unittest.main()
