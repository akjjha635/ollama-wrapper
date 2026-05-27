import unittest

from ollama_wrapper.eval import BenchmarkRunner, BenchmarkSample


class TestEvalBenchmarks(unittest.TestCase):
    def test_summarize_samples(self):
        samples = [
            BenchmarkSample(latency_ms=10.0, status_code=200, input_tokens=2, output_tokens=3, total_tokens=5),
            BenchmarkSample(latency_ms=20.0, status_code=200, input_tokens=4, output_tokens=5, total_tokens=9),
            BenchmarkSample(latency_ms=30.0, status_code=500, input_tokens=0, output_tokens=0, total_tokens=0),
        ]

        report = BenchmarkRunner.summarize(samples)
        self.assertEqual(report.request_count, 3)
        self.assertEqual(report.success_count, 2)
        self.assertEqual(report.error_count, 1)
        self.assertAlmostEqual(report.avg_latency_ms, 20.0)
        self.assertGreater(report.p95_latency_ms, 0.0)
        self.assertAlmostEqual(report.avg_total_tokens, (5 + 9 + 0) / 3.0)

    def test_compare_reports(self):
        baseline = BenchmarkRunner.summarize(
            [BenchmarkSample(latency_ms=10.0, status_code=200, input_tokens=1, output_tokens=2, total_tokens=3)]
        )
        candidate = BenchmarkRunner.summarize(
            [BenchmarkSample(latency_ms=12.0, status_code=200, input_tokens=2, output_tokens=2, total_tokens=4)]
        )

        comparison = BenchmarkRunner.compare_reports(
            baseline,
            candidate,
            baseline_label="linear",
            candidate_label="faiss",
        )

        self.assertEqual(comparison.baseline_label, "linear")
        self.assertEqual(comparison.candidate_label, "faiss")
        self.assertAlmostEqual(comparison.deltas["avg_latency_ms"], 2.0)
        self.assertAlmostEqual(comparison.deltas["avg_total_tokens"], 1.0)


if __name__ == "__main__":
    unittest.main()
