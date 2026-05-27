import unittest

from ollama_wrapper.control import GovernanceConfig, GovernancePolicy
from ollama_wrapper.llm import LLMChatResult, LLMMessage, LLMProvider, LLMUsage
from ollama_wrapper.orchestration import DefaultQueryOrchestrator, QueryRequest


class DummyProvider(LLMProvider):
    def provider_name(self) -> str:
        return "dummy"

    def chat(self, model, messages, response_schema=None, options=None):
        text = " | ".join([m.content for m in messages])
        return LLMChatResult(content=f"sync:{text}", model=model, usage=LLMUsage(total_tokens=5), raw={})

    async def chat_async(self, model, messages, response_schema=None, options=None):
        text = " | ".join([m.content for m in messages])
        return LLMChatResult(content=f"async:{text}", model=model, usage=LLMUsage(total_tokens=7), raw={})

    def embed(self, model, input_text, options=None):
        return [0.1, 0.2]

    async def embed_async(self, model, input_text, options=None):
        return [0.1, 0.2]

    def health_check(self) -> bool:
        return True


class StreamingDummyProvider(DummyProvider):
    async def chat_stream_async(self, model, messages, response_schema=None, options=None):
        _ = model
        _ = messages
        _ = response_schema
        _ = options
        for chunk in ["hello", " ", "stream"]:
            yield chunk


class TestOrchestrationContracts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        provider = DummyProvider()
        self.orchestrator = DefaultQueryOrchestrator.from_single_provider(
            provider_key="dummy",
            provider=provider,
            model="dummy-model",
        )

    def test_run_query_sync(self):
        req = QueryRequest(
            user_query="hello",
            system_prompt="be concise",
            options={
                "token_budget": 40,
                "rag_candidates": [
                    {"text": "short candidate", "semantic_score": 0.6, "lexical_score": 0.5},
                    {"text": "another candidate", "semantic_score": 0.2, "lexical_score": 0.3},
                ],
            },
        )
        result = self.orchestrator.run_query(req)
        self.assertEqual(result.provider, "dummy")
        self.assertEqual(result.model, "dummy-model")
        self.assertIn("sync:", result.reply)

    def test_preview_route_contains_optimization(self):
        req = QueryRequest(
            user_query="hello",
            options={
                "token_budget": 30,
                "rag_candidates": [
                    {"text": "alpha context", "semantic_score": 0.9, "lexical_score": 0.1},
                    {"text": "beta context", "semantic_score": 0.1, "lexical_score": 0.9},
                ],
            },
        )
        preview = self.orchestrator.preview_route(req)
        self.assertEqual(preview.provider, "dummy")
        self.assertIn("candidate_count", preview.optimization)
        self.assertEqual(preview.optimization["candidate_count"], 2)
        self.assertEqual(preview.route_explanation.get("strategy"), "policy-routing-v1")
        self.assertIn("query_type", preview.session_plan)
        self.assertIn("confidence_scores", preview.optimization)

    def test_route_explanation_and_session_plan_present(self):
        req = QueryRequest(user_query="why does this fail", options={"session_turn_count": 3})
        result = self.orchestrator.run_query(req)
        self.assertEqual(result.route_explanation.get("strategy"), "policy-routing-v1")
        self.assertEqual(result.session_plan.get("query_type"), "reasoning")
        self.assertTrue(result.session_plan.get("include_summary"))

    def test_guardrails_rejects_disallowed_route(self):
        provider = DummyProvider()
        guarded = DefaultQueryOrchestrator.from_single_provider(
            provider_key="dummy",
            provider=provider,
            model="dummy-model",
        )
        guarded.guardrails_policy = GovernancePolicy(
            GovernanceConfig(
                allowed_providers={"dummy"},
                allowed_models={"other-model"},
                max_payload_chars=200,
            )
        )
        with self.assertRaises(ValueError):
            guarded.run_query(QueryRequest(user_query="hello"))

    async def test_run_query_async(self):
        req = QueryRequest(user_query="hello async", system_prompt="be concise")
        result = await self.orchestrator.run_query_async(req)
        self.assertEqual(result.provider, "dummy")
        self.assertEqual(result.model, "dummy-model")
        self.assertIn("async:", result.reply)

    async def test_run_query_stream_async(self):
        provider = StreamingDummyProvider()
        orchestrator = DefaultQueryOrchestrator.from_single_provider(
            provider_key="dummy",
            provider=provider,
            model="dummy-model",
        )
        meta, stream = await orchestrator.run_query_stream_async(QueryRequest(user_query="hello"))
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        self.assertEqual(meta.provider, "dummy")
        self.assertEqual("".join(chunks), "hello stream")


if __name__ == "__main__":
    unittest.main()
