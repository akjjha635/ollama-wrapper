import unittest
import tempfile

from ollama_wrapper.control import (
    GovernanceConfig,
    GovernancePolicy,
    InMemoryRateLimiter,
    SQLiteRateLimiter,
    TenantGovernanceRule,
)
from ollama_wrapper.orchestration import QueryRequest, RoutingDecision


class TestControlLayer(unittest.TestCase):
    def test_rate_limiter_allows_then_rejects_burst(self):
        limiter = InMemoryRateLimiter(default_qps=1.0, default_burst=1)
        first = limiter.allow(session_id="s1", endpoint="/message", options={})
        second = limiter.allow(session_id="s1", endpoint="/message", options={})
        self.assertTrue(first.allowed)
        self.assertFalse(second.allowed)
        self.assertGreaterEqual(second.retry_after_ms, 1)

    def test_governance_policy_rejects_payload_size(self):
        policy = GovernancePolicy(
            GovernanceConfig(
                allowed_providers={"dummy"},
                allowed_models={"m1"},
                max_payload_chars=5,
            )
        )
        req = QueryRequest(user_query="this is too long")
        route = RoutingDecision(provider="dummy", model="m1", reason="test")
        decision = policy.evaluate(req, route)
        self.assertEqual(decision.action, "reject")
        self.assertEqual(decision.reason, "payload-too-large")

    def test_sqlite_rate_limiter_allows_then_rejects_burst(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            limiter = SQLiteRateLimiter(db_path=tmp.name, default_qps=1.0, default_burst=1)
            first = limiter.allow(session_id="s1", endpoint="/message", options={})
            second = limiter.allow(session_id="s1", endpoint="/message", options={})
            self.assertTrue(first.allowed)
            self.assertFalse(second.allowed)

    def test_tenant_override_blocks_model(self):
        policy = GovernancePolicy(
            GovernanceConfig(
                allowed_providers={"dummy"},
                allowed_models={"m1", "m2"},
                max_payload_chars=100,
                tenant_overrides={
                    "tenant-a": TenantGovernanceRule(
                        allowed_models={"m1"},
                    )
                },
            )
        )
        req = QueryRequest(user_query="ok", options={"tenant_id": "tenant-a"})
        route = RoutingDecision(provider="dummy", model="m2", reason="test")
        decision = policy.evaluate(req, route)
        self.assertEqual(decision.action, "reject")
        self.assertEqual(decision.reason, "model-not-allowed")


if __name__ == "__main__":
    unittest.main()
