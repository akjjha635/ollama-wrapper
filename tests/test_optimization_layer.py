import unittest

from ollama_wrapper.optimization import MathematicalOptimizationLayer, OptimizationConfig, RAGCandidate


class TestOptimizationLayer(unittest.TestCase):
    def test_optimize_context_selects_by_weighted_score_and_budget(self):
        layer = MathematicalOptimizationLayer()
        candidates = [
            RAGCandidate(text="alpha one two", semantic_score=0.9, lexical_score=0.1),
            RAGCandidate(text="beta one", semantic_score=0.2, lexical_score=0.8),
            RAGCandidate(text="gamma one two three four five", semantic_score=0.7, lexical_score=0.6),
        ]
        config = OptimizationConfig(
            semantic_weight=0.7,
            lexical_weight=0.3,
            token_budget=12,
            max_context_items=2,
        )

        result = layer.optimize_context(candidates, config)
        self.assertLessEqual(result.token_estimate, 12)
        self.assertLessEqual(len(result.selected_indices), 2)
        self.assertGreaterEqual(len(result.selected_indices), 1)

    def test_parse_candidates_ignores_invalid_entries(self):
        layer = MathematicalOptimizationLayer()
        raw = [
            {"text": "valid", "semantic_score": 1.0, "lexical_score": 0.0},
            {"text": "   "},
            "invalid",
        ]
        parsed = layer.parse_candidates(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].text, "valid")

    def test_compute_bm25_score_returns_positive_for_matching_terms(self):
        layer = MathematicalOptimizationLayer()
        query_terms = ["alpha", "beta"]
        doc_terms = ["alpha", "alpha", "gamma", "beta"]
        score = layer.compute_bm25_score(
            query_terms=query_terms,
            doc_terms=doc_terms,
            doc_len=len(doc_terms),
            avg_doc_len=4.0,
            term_df={"alpha": 2, "beta": 3},
            total_docs=10,
            k1=1.5,
            b=0.75,
        )
        self.assertGreater(score, 0.0)

    def test_diversity_aware_packing_avoids_same_source(self):
        layer = MathematicalOptimizationLayer()
        candidates = [
            RAGCandidate(text="alpha one", semantic_score=0.9, lexical_score=0.9, metadata={"source": "doc-a"}),
            RAGCandidate(text="alpha two", semantic_score=0.89, lexical_score=0.89, metadata={"source": "doc-a"}),
            RAGCandidate(text="beta one", semantic_score=0.85, lexical_score=0.85, metadata={"source": "doc-b"}),
        ]
        config = OptimizationConfig(token_budget=50, max_context_items=2, diversity_lambda=0.5)
        result = layer.optimize_context(candidates, config)
        selected_sources = [candidates[i].metadata.get("source") for i in result.selected_indices]
        self.assertIn("doc-a", selected_sources)
        self.assertIn("doc-b", selected_sources)

    def test_query_type_calibration_and_confidence_scores(self):
        layer = MathematicalOptimizationLayer()
        candidates = [
            RAGCandidate(text="one", semantic_score=0.2, lexical_score=0.9),
            RAGCandidate(text="two", semantic_score=0.9, lexical_score=0.2),
        ]
        factoid = layer.optimize_context(candidates, OptimizationConfig(query_type="factoid", token_budget=20, max_context_items=2))
        reasoning = layer.optimize_context(candidates, OptimizationConfig(query_type="reasoning", token_budget=20, max_context_items=2))
        self.assertNotEqual(factoid.selected_indices, reasoning.selected_indices)
        self.assertEqual(len(factoid.confidence_scores), len(factoid.selected_indices))
        if factoid.confidence_scores:
            self.assertAlmostEqual(sum(factoid.confidence_scores), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
