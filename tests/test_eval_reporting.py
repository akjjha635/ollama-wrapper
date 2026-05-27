import unittest

from ollama_wrapper.eval import comparison_to_markdown


class TestEvalReporting(unittest.TestCase):
    def test_comparison_to_markdown_contains_table(self):
        payload = {
            "baseline_label": "linear",
            "candidate_label": "faiss",
            "baseline": {
                "request_count": 10,
                "success_count": 10,
                "error_count": 0,
                "avg_latency_ms": 100.0,
                "p50_latency_ms": 95.0,
                "p95_latency_ms": 130.0,
                "avg_input_tokens": 10.0,
                "avg_output_tokens": 20.0,
                "avg_total_tokens": 30.0,
            },
            "candidate": {
                "request_count": 10,
                "success_count": 10,
                "error_count": 0,
                "avg_latency_ms": 90.0,
                "p50_latency_ms": 85.0,
                "p95_latency_ms": 120.0,
                "avg_input_tokens": 10.0,
                "avg_output_tokens": 20.0,
                "avg_total_tokens": 30.0,
            },
            "deltas": {
                "request_count": 0.0,
                "success_count": 0.0,
                "error_count": 0.0,
                "avg_latency_ms": -10.0,
                "p50_latency_ms": -10.0,
                "p95_latency_ms": -10.0,
                "avg_input_tokens": 0.0,
                "avg_output_tokens": 0.0,
                "avg_total_tokens": 0.0,
            },
        }

        md = comparison_to_markdown(payload)
        self.assertIn("## Benchmark Comparison", md)
        self.assertIn("| Metric | linear | faiss |", md)
        self.assertIn("| avg_latency_ms |", md)


if __name__ == "__main__":
    unittest.main()
