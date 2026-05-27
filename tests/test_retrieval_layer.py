import unittest

import numpy as np

from ollama_wrapper.optimization import MathematicalOptimizationLayer
from ollama_wrapper.retrieval import (
    DenseBackend,
    DenseBackendRegistry,
    FirstCandidateReranker,
    HybridRetriever,
)


class _DummyEmbedResponse:
    def __init__(self, vector):
        self.embeddings = [vector]


class _DummyGenerateResponse:
    def __init__(self, text: str):
        self.response = text


class _DummySyncClient:
    def embed(self, model, input):
        _ = model
        _ = input
        return _DummyEmbedResponse([1.0, 0.0])

    def generate(self, model, prompt):
        _ = model
        _ = prompt
        return _DummyGenerateResponse("0")


class _DummyWrapper:
    def __init__(self):
        self.embed_model = "dummy-embed"
        self.llm_model = "dummy-llm"
        self._sync_client = _DummySyncClient()
        self._async_client = None
        self._optimization_layer = MathematicalOptimizationLayer()
        self.k1 = 1.5
        self.b = 0.75

        self.vector_database = [
            {"text": "alpha signal", "vector": np.array([1.0, 0.0], dtype=np.float32), "metadata": {"scope": "a"}},
            {"text": "beta signal", "vector": np.array([0.0, 1.0], dtype=np.float32), "metadata": {"scope": "b"}},
        ]
        self.doc_lens = [2, 2]
        self.avg_doc_len = 2.0
        self.df = {"alpha": 1, "signal": 2, "beta": 1}
        self.warnings = []

    def _strip_thinking_tags(self, text: str) -> str:
        return text

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _CustomDenseBackend(DenseBackend):
    def score(self, matrix, query_vector, warn):
        _ = query_vector
        _ = warn
        # Strongly bias first candidate deterministically.
        return np.array([1.0] + [0.0 for _ in range(max(0, matrix.shape[0] - 1))], dtype=np.float32)


class TestRetrievalLayer(unittest.TestCase):
    def test_retrieve_and_rerank_with_metadata_filter(self):
        wrapper = _DummyWrapper()
        retriever = HybridRetriever(wrapper)

        result = retriever.retrieve_and_rerank(
            query="alpha",
            metadata_filter={"scope": "a"},
            sync_mode=True,
        )
        self.assertEqual(result, "alpha signal")

    def test_retrieve_and_rerank_with_custom_reranker(self):
        wrapper = _DummyWrapper()
        retriever = HybridRetriever(wrapper, reranker=FirstCandidateReranker())

        result = retriever.retrieve_and_rerank(query="alpha", metadata_filter=None, sync_mode=True)
        self.assertEqual(result, "alpha signal")

    def test_unknown_backend_falls_back_to_linear(self):
        wrapper = _DummyWrapper()
        retriever = HybridRetriever(wrapper, backend="unknown-backend", reranker=FirstCandidateReranker())

        result = retriever.retrieve_and_rerank(query="alpha", metadata_filter=None, sync_mode=True)
        self.assertEqual(result, "alpha signal")

    def test_custom_backend_object_is_supported(self):
        wrapper = _DummyWrapper()
        retriever = HybridRetriever(wrapper, backend=_CustomDenseBackend(), reranker=FirstCandidateReranker())

        result = retriever.retrieve_and_rerank(query="alpha", metadata_filter=None, sync_mode=True)
        self.assertEqual(result, "alpha signal")

    def test_custom_backend_registry_is_supported(self):
        wrapper = _DummyWrapper()
        registry = DenseBackendRegistry()
        registry.register("my-backend", _CustomDenseBackend)
        retriever = HybridRetriever(
            wrapper,
            backend="my-backend",
            reranker=FirstCandidateReranker(),
            backend_registry=registry,
        )

        result = retriever.retrieve_and_rerank(query="alpha", metadata_filter=None, sync_mode=True)
        self.assertEqual(result, "alpha signal")


if __name__ == "__main__":
    unittest.main()
