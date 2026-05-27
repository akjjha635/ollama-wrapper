import asyncio

from fastapi.testclient import TestClient
from ollama_wrapper.api_server import ChatSessionManager, create_app
from ollama_wrapper.control import GovernanceConfig, GovernancePolicy, InMemoryRateLimiter, TokenBudgetPolicy
from ollama_wrapper.llm import LLMChatResult, LLMProvider, LLMUsage
from ollama_wrapper.orchestration import DefaultQueryOrchestrator, QueryRequest, StaticRoutingPolicy


class DummyWrapper:
    async def ask_async(self, message, metadata_filter=None):
        return f"echo: {message}"


class DummyProvider(LLMProvider):
    def provider_name(self) -> str:
        return "dummy"

    def chat(self, model, messages, response_schema=None, options=None):
        text = " ".join(m.content for m in messages)
        return LLMChatResult(content=f"sync:{text}", model=model, usage=LLMUsage(total_tokens=11), raw={})

    async def chat_async(self, model, messages, response_schema=None, options=None):
        text = " ".join(m.content for m in messages)
        return LLMChatResult(content=f"async:{text}", model=model, usage=LLMUsage(total_tokens=13), raw={})

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
        for chunk in ["chunk-1", "chunk-2"]:
            yield chunk


class FailingStreamingDummyProvider(DummyProvider):
    async def chat_stream_async(self, model, messages, response_schema=None, options=None):
        _ = model
        _ = messages
        _ = response_schema
        _ = options
        yield "partial"
        raise RuntimeError("stream failure")


def test_session_create_and_message():
    wrapper = DummyWrapper()
    manager = ChatSessionManager(wrapper)
    app = create_app(manager)

    client = TestClient(app)

    # create session
    resp = client.post("/session", json={"system_prompt": "You are helpful."})
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    # send message
    r2 = client.post(f"/session/{sid}/message", json={"message": "hello"})
    assert r2.status_code == 200
    assert r2.json()["reply"] == "echo: hello"

    # get state
    r3 = client.get(f"/session/{sid}")
    assert r3.status_code == 200
    state = r3.json()
    assert state["session_id"] == sid
    assert any(m["content"] == "hello" for m in state["chat_history"]) 


def test_session_message_with_orchestrator_metadata():
    provider = DummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
    )
    manager = ChatSessionManager(wrapper=None, orchestrator=orchestrator)
    app = create_app(manager)

    client = TestClient(app)

    resp = client.post("/session", json={"system_prompt": "You are helpful."})
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    r2 = client.post(
        f"/session/{sid}/message",
        json={"message": "hello", "options": {"budget_tier": "low"}},
    )
    assert r2.status_code == 200
    payload = r2.json()
    assert payload["reply"].startswith("async:")
    assert payload["provider"] == "dummy"
    assert payload["model"] == "dummy-model"
    assert payload["route_reason"] == "static-route"
    assert payload["route_explanation"]["strategy"] == "policy-routing-v1"
    assert payload["session_plan"]["query_type"] in {"general", "factoid", "reasoning", "code"}
    assert payload["guardrails"]["action"] in {"allow", "reject"}
    assert payload["usage"]["total_tokens"] == 13
    assert payload["budget_decision"]["status"] == "not-enforced"


def test_session_dry_run_with_orchestrator():
    provider = DummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
    )
    manager = ChatSessionManager(wrapper=None, orchestrator=orchestrator)
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful.", "metadata": {"team": "ml"}})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    dry = client.post(
        f"/session/{sid}/dry-run",
        json={
            "message": "hello",
            "options": {
                "budget_tier": "low",
                "cheap_model": "cheap-v1",
                "token_budget": 30,
                "rag_candidates": [
                    {"text": "candidate one", "semantic_score": 0.8, "lexical_score": 0.2},
                    {"text": "candidate two", "semantic_score": 0.4, "lexical_score": 0.6},
                ],
            },
        },
    )
    assert dry.status_code == 200
    payload = dry.json()
    assert payload["dry_run"] is True
    assert payload["provider"] == "dummy"
    assert payload["model"] == "dummy-model"
    assert payload["route_reason"] == "static-route"
    assert payload["route_explanation"]["strategy"] == "policy-routing-v1"
    assert payload["session_plan"]["query_type"] in {"general", "factoid", "reasoning", "code"}
    assert payload["guardrails"]["action"] in {"allow", "reject"}
    assert payload["context_plan"]["include_metadata_filter"] is True
    assert payload["optimization"]["candidate_count"] == 2
    assert payload["optimization"]["selected_count"] >= 1
    assert "confidence_scores" in payload["optimization"]
    assert payload["budget"]["status"] == "not-enforced"


def test_session_dry_run_with_wrapper_fallback():
    manager = ChatSessionManager(wrapper=DummyWrapper(), orchestrator=None)
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful."})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    dry = client.post(f"/session/{sid}/dry-run", json={"message": "hello"})
    assert dry.status_code == 200
    payload = dry.json()
    assert payload["dry_run"] is True
    assert payload["provider"] == "wrapper"
    assert payload["route_reason"] == "wrapper-fallback"


def test_session_dry_run_missing_session_returns_404():
    manager = ChatSessionManager(wrapper=DummyWrapper())
    app = create_app(manager)
    client = TestClient(app)

    dry = client.post("/session/missing/dry-run", json={"message": "hello"})
    assert dry.status_code == 404


def test_debug_traces_and_metrics_endpoints():
    provider = DummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
    )
    manager = ChatSessionManager(wrapper=None, orchestrator=orchestrator)
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful."})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    msg = client.post(f"/session/{sid}/message", json={"message": "hello"})
    assert msg.status_code == 200

    dry = client.post(f"/session/{sid}/dry-run", json={"message": "hello"})
    assert dry.status_code == 200

    traces = client.get("/debug/traces", params={"limit": 10})
    assert traces.status_code == 200
    traces_payload = traces.json()
    assert traces_payload["count"] >= 2
    assert any(t["endpoint"] == "/session/{session_id}/message" for t in traces_payload["traces"])
    assert any(t["endpoint"] == "/session/{session_id}/dry-run" for t in traces_payload["traces"])

    metrics = client.get("/debug/metrics")
    assert metrics.status_code == 200
    metrics_payload = metrics.json()
    assert metrics_payload["request_count"] >= 2
    assert metrics_payload["dry_run_count"] >= 1


def test_session_message_over_budget_truncate_mode():
    provider = DummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
        budget_policy=TokenBudgetPolicy(default_max_input_tokens=8, default_mode="truncate"),
    )
    manager = ChatSessionManager(wrapper=None, orchestrator=orchestrator)
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful."})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    long_message = " ".join(["token"] * 40)
    r2 = client.post(
        f"/session/{sid}/message",
        json={"message": long_message, "options": {"budget_mode": "truncate", "max_input_tokens": 8}},
    )
    assert r2.status_code == 200
    payload = r2.json()
    assert payload["budget_decision"]["action"] == "truncate"
    assert payload["budget_decision"]["status"] == "over-budget"
    assert payload["budget_decision"]["effective_input_tokens"] <= 8


def test_session_message_over_budget_reject_mode_returns_400():
    provider = DummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
        budget_policy=TokenBudgetPolicy(default_max_input_tokens=8, default_mode="reject"),
    )
    manager = ChatSessionManager(wrapper=None, orchestrator=orchestrator)
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful."})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    long_message = " ".join(["token"] * 40)
    r2 = client.post(
        f"/session/{sid}/message",
        json={"message": long_message, "options": {"budget_mode": "reject", "max_input_tokens": 8}},
    )
    assert r2.status_code == 400
    assert "exceeds input token budget" in r2.json()["detail"]


def test_session_dry_run_over_budget_reject_mode_returns_preview():
    provider = DummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
        budget_policy=TokenBudgetPolicy(default_max_input_tokens=8, default_mode="reject"),
    )
    manager = ChatSessionManager(wrapper=None, orchestrator=orchestrator)
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful."})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    long_message = " ".join(["token"] * 40)
    dry = client.post(
        f"/session/{sid}/dry-run",
        json={"message": long_message, "options": {"budget_mode": "reject", "max_input_tokens": 8}},
    )
    assert dry.status_code == 200
    payload = dry.json()
    assert payload["budget"]["action"] == "reject"
    assert payload["budget"]["status"] == "over-budget"


def test_message_rate_limit_returns_429():
    provider = DummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
    )
    manager = ChatSessionManager(
        wrapper=None,
        orchestrator=orchestrator,
        rate_limiter=InMemoryRateLimiter(default_qps=1.0, default_burst=1),
    )
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful."})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    first = client.post(f"/session/{sid}/message", json={"message": "hello"})
    assert first.status_code == 200

    second = client.post(f"/session/{sid}/message", json={"message": "hello again"})
    assert second.status_code == 429
    detail = second.json()["detail"]
    assert detail["message"] == "Rate limit exceeded"
    assert detail["retry_after_ms"] >= 1


def test_governance_rejects_disallowed_model_message_and_allows_dry_run_preview():
    provider = DummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
        guardrails_policy=GovernancePolicy(
            GovernanceConfig(
                allowed_providers={"dummy"},
                allowed_models={"other-model"},
                max_payload_chars=1000,
            )
        ),
    )
    manager = ChatSessionManager(wrapper=None, orchestrator=orchestrator)
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful."})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    dry = client.post(f"/session/{sid}/dry-run", json={"message": "hello"})
    assert dry.status_code == 200
    assert dry.json()["guardrails"]["action"] == "reject"

    message = client.post(f"/session/{sid}/message", json={"message": "hello"})
    assert message.status_code == 400
    assert "Guardrails rejected request" in message.json()["detail"]


def test_stream_endpoint_returns_chunked_text():
    provider = StreamingDummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
    )
    manager = ChatSessionManager(wrapper=None, orchestrator=orchestrator)
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful."})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    with client.stream("POST", f"/session/{sid}/stream", json={"message": "hello stream"}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "chunk-1chunk-2" in body


def test_stream_failure_records_non_200_trace_and_no_assistant_turn():
    provider = FailingStreamingDummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
    )
    manager = ChatSessionManager(wrapper=None, orchestrator=orchestrator)
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful."})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    stream_failed = False
    try:
        with client.stream("POST", f"/session/{sid}/stream", json={"message": "hello stream"}) as response:
            assert response.status_code == 200
            _ = "".join(response.iter_text())
    except Exception:
        stream_failed = True
    assert stream_failed

    traces = client.get("/debug/traces", params={"limit": 10})
    assert traces.status_code == 200
    trace_items = traces.json()["traces"]
    stream_traces = [t for t in trace_items if t["endpoint"] == "/session/{session_id}/stream"]
    assert stream_traces
    assert any(t["status_code"] == 500 for t in stream_traces)

    state = client.get(f"/session/{sid}")
    assert state.status_code == 200
    history = state.json()["chat_history"]
    assert len(history) == 1
    assert history[0]["role"] == "user"


def test_rejected_message_does_not_append_user_turn():
    provider = DummyProvider()
    orchestrator = DefaultQueryOrchestrator(
        providers={"dummy": provider},
        routing_policy=StaticRoutingPolicy(provider="dummy", model="dummy-model"),
        guardrails_policy=GovernancePolicy(
            GovernanceConfig(
                allowed_providers={"dummy"},
                allowed_models={"other-model"},
                max_payload_chars=1000,
            )
        ),
    )
    manager = ChatSessionManager(wrapper=None, orchestrator=orchestrator)
    app = create_app(manager)
    client = TestClient(app)

    created = client.post("/session", json={"system_prompt": "You are helpful."})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    message = client.post(f"/session/{sid}/message", json={"message": "hello"})
    assert message.status_code == 400

    state = client.get(f"/session/{sid}")
    assert state.status_code == 200
    assert state.json()["chat_history"] == []
