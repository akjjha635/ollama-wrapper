import asyncio
import time
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .control import InMemoryRateLimiter, RateLimiter
from .observability import MetricsCollector, RequestTrace, RequestTraceStore
from .orchestration import QueryOrchestrator, QueryRequest


class SessionCreateRequest(BaseModel):
    system_prompt: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = None


class MessageRequest(BaseModel):
    message: str
    options: Optional[Dict[str, Any]] = None


class DryRunRequest(BaseModel):
    message: str
    options: Optional[Dict[str, Any]] = None


class StreamRequest(BaseModel):
    message: str
    options: Optional[Dict[str, Any]] = None


class ChatSession:
    def __init__(self, session_id: str, system_prompt: str = "", metadata: dict = None):
        self.session_id = session_id
        self.system_prompt = system_prompt or ""
        self.metadata = metadata or {}
        self.chat_history = []
        self.running_summary = ""
        self.created_at = asyncio.get_event_loop().time()


class ChatSessionManager:
    """Manage chat sessions and per-session context for multiple agents/connections."""

    def __init__(
        self,
        wrapper=None,
        orchestrator: Optional[QueryOrchestrator] = None,
        rate_limiter: Optional[RateLimiter] = None,
        trace_store: Optional[RequestTraceStore] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self.wrapper = wrapper
        self.orchestrator = orchestrator
        self.rate_limiter = rate_limiter or InMemoryRateLimiter()
        self.trace_store = trace_store or RequestTraceStore()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.sessions: Dict[str, ChatSession] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.lock = asyncio.Lock()

    def get_or_create_session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self.session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self.session_locks[session_id] = lock
        return lock

    @staticmethod
    def _with_session_options(options: Optional[Dict[str, Any]], sess: ChatSession) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(options or {})
        merged.setdefault("session_turn_count", len(sess.chat_history) // 2)
        tenant_from_metadata = (sess.metadata or {}).get("tenant_id")
        if tenant_from_metadata and "tenant_id" not in merged:
            merged["tenant_id"] = tenant_from_metadata
        return merged

    async def create_session(self, system_prompt: str = "", metadata: dict = None) -> ChatSession:
        async with self.lock:
            sid = str(uuid.uuid4())
            sess = ChatSession(sid, system_prompt=system_prompt, metadata=metadata)
            self.sessions[sid] = sess
            self.session_locks[sid] = asyncio.Lock()
            return sess

    async def get_session(self, session_id: str) -> ChatSession:
        sess = self.sessions.get(session_id)
        if not sess:
            raise KeyError(f"Session {session_id} not found")
        return sess

    async def close_session(self, session_id: str) -> None:
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.session_locks:
                del self.session_locks[session_id]


def create_app(manager: ChatSessionManager) -> FastAPI:
    app = FastAPI()

    @app.post("/session")
    async def create(req: SessionCreateRequest):
        sess = await manager.create_session(system_prompt=req.system_prompt, metadata=req.metadata)
        return {"session_id": sess.session_id}

    @app.post("/session/{session_id}/message")
    async def message(session_id: str, req: MessageRequest):
        start = time.perf_counter()
        session_lock = manager.get_or_create_session_lock(session_id)
        async with session_lock:
            try:
                sess = await manager.get_session(session_id)
            except KeyError:
                raise HTTPException(status_code=404, detail="session not found")

            merged_options = manager._with_session_options(req.options, sess)

            rate_limit = manager.rate_limiter.allow(
                session_id=session_id,
                endpoint="/session/{session_id}/message",
                options=merged_options,
            )
            if not rate_limit.allowed:
                status_code = 429
                latency_ms = (time.perf_counter() - start) * 1000.0
                manager.trace_store.add(
                    RequestTrace(
                        timestamp_ms=int(time.time() * 1000),
                        endpoint="/session/{session_id}/message",
                        session_id=session_id,
                        provider="rate-limiter",
                        model="none",
                        route_reason="rate-limit",
                        status_code=status_code,
                        latency_ms=latency_ms,
                        usage={},
                        extra={
                            "rate_limit": {
                                "qps": rate_limit.qps,
                                "burst": rate_limit.burst,
                                "retry_after_ms": rate_limit.retry_after_ms,
                                "reason": rate_limit.reason,
                            }
                        },
                    )
                )
                manager.metrics_collector.record(latency_ms=latency_ms, status_code=status_code)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "message": "Rate limit exceeded",
                        "retry_after_ms": rate_limit.retry_after_ms,
                        "reason": rate_limit.reason,
                    },
                )

            # Use the provided OllamaWrapper instance to ask asynchronously
            provider = "wrapper"
            model = getattr(manager.wrapper, "llm_model", "unknown") if manager.wrapper else "unknown"
            usage: Dict[str, Any] = {}
            route_reason = ""
            route_explanation: Dict[str, Any] = {}
            budget_decision: Dict[str, Any] = {}
            guardrails: Dict[str, Any] = {}
            session_plan: Dict[str, Any] = {}
            status_code = 200
            try:
                if manager.orchestrator is not None:
                    query_response = await manager.orchestrator.run_query_async(
                        QueryRequest(
                            user_query=req.message,
                            session_id=session_id,
                            system_prompt=sess.system_prompt,
                            metadata_filter=sess.metadata,
                            options=merged_options,
                        )
                    )
                    resp = query_response.reply
                    provider = query_response.provider
                    model = query_response.model
                    usage = query_response.usage
                    route_reason = query_response.route_reason
                    route_explanation = query_response.route_explanation
                    budget_decision = query_response.budget_decision
                    guardrails = query_response.guardrails
                    session_plan = query_response.session_plan
                elif manager.wrapper is not None and hasattr(manager.wrapper, "ask_async"):
                    resp = await manager.wrapper.ask_async(req.message, metadata_filter=sess.metadata)
                    budget_decision = {}
                else:
                    # fallback to sync ask
                    loop = asyncio.get_running_loop()
                    resp = await loop.run_in_executor(None, manager.wrapper.ask, req.message, sess.metadata)
                    budget_decision = {}
            except ValueError as e:
                status_code = 400
                latency_ms = (time.perf_counter() - start) * 1000.0
                manager.trace_store.add(
                    RequestTrace(
                        timestamp_ms=int(time.time() * 1000),
                        endpoint="/session/{session_id}/message",
                        session_id=session_id,
                        provider=provider,
                        model=model,
                        route_reason=route_reason,
                        status_code=status_code,
                        latency_ms=latency_ms,
                        budget_action=str(budget_decision.get("action", "")),
                        budget_status=str(budget_decision.get("status", "")),
                        budget_reason=str(budget_decision.get("reason", "")),
                        usage=usage,
                    )
                )
                manager.metrics_collector.record(latency_ms=latency_ms, status_code=status_code)
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                status_code = 500
                latency_ms = (time.perf_counter() - start) * 1000.0
                manager.trace_store.add(
                    RequestTrace(
                        timestamp_ms=int(time.time() * 1000),
                        endpoint="/session/{session_id}/message",
                        session_id=session_id,
                        provider=provider,
                        model=model,
                        route_reason=route_reason,
                        status_code=status_code,
                        latency_ms=latency_ms,
                        budget_action=str(budget_decision.get("action", "")),
                        budget_status=str(budget_decision.get("status", "")),
                        budget_reason=str(budget_decision.get("reason", "")),
                        usage=usage,
                    )
                )
                manager.metrics_collector.record(latency_ms=latency_ms, status_code=status_code)
                raise HTTPException(status_code=500, detail=str(e))

            # Keep turn consistency by adding user+assistant only on successful completion.
            sess.chat_history.append({"role": "user", "content": req.message})
            sess.chat_history.append({"role": "assistant", "content": resp})

            payload = {
                "reply": resp,
                "provider": provider,
                "model": model,
                "usage": usage,
                "route_reason": route_reason,
                "route_explanation": route_explanation,
                "budget_decision": budget_decision,
                "guardrails": guardrails,
                "session_plan": session_plan,
            }

            latency_ms = (time.perf_counter() - start) * 1000.0
            manager.trace_store.add(
                RequestTrace(
                    timestamp_ms=int(time.time() * 1000),
                    endpoint="/session/{session_id}/message",
                    session_id=session_id,
                    provider=provider,
                    model=model,
                    route_reason=route_reason,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    budget_action=str(budget_decision.get("action", "")),
                    budget_status=str(budget_decision.get("status", "")),
                    budget_reason=str(budget_decision.get("reason", "")),
                    usage=usage,
                )
            )
            manager.metrics_collector.record(
                latency_ms=latency_ms,
                status_code=status_code,
                input_tokens=(usage or {}).get("input_tokens"),
                output_tokens=(usage or {}).get("output_tokens"),
                total_tokens=(usage or {}).get("total_tokens"),
            )
            return payload

    @app.post("/session/{session_id}/dry-run")
    async def dry_run(session_id: str, req: DryRunRequest):
        start = time.perf_counter()
        try:
            sess = await manager.get_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found")

        merged_options = manager._with_session_options(req.options, sess)

        if manager.orchestrator is not None:
            rate_limit = manager.rate_limiter.allow(
                session_id=session_id,
                endpoint="/session/{session_id}/dry-run",
                options=merged_options,
            )
            if not rate_limit.allowed:
                status_code = 429
                latency_ms = (time.perf_counter() - start) * 1000.0
                manager.trace_store.add(
                    RequestTrace(
                        timestamp_ms=int(time.time() * 1000),
                        endpoint="/session/{session_id}/dry-run",
                        session_id=session_id,
                        provider="rate-limiter",
                        model="none",
                        route_reason="rate-limit",
                        status_code=status_code,
                        latency_ms=latency_ms,
                        usage={},
                        extra={
                            "rate_limit": {
                                "qps": rate_limit.qps,
                                "burst": rate_limit.burst,
                                "retry_after_ms": rate_limit.retry_after_ms,
                                "reason": rate_limit.reason,
                            }
                        },
                    )
                )
                manager.metrics_collector.record(latency_ms=latency_ms, status_code=status_code, is_dry_run=True)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "message": "Rate limit exceeded",
                        "retry_after_ms": rate_limit.retry_after_ms,
                        "reason": rate_limit.reason,
                    },
                )

            preview = manager.orchestrator.preview_route(
                QueryRequest(
                    user_query=req.message,
                    session_id=session_id,
                    system_prompt=sess.system_prompt,
                    metadata_filter=sess.metadata,
                    options=merged_options,
                )
            )
            payload = {
                "dry_run": True,
                "provider": preview.provider,
                "model": preview.model,
                "route_reason": preview.route_reason,
                "route_explanation": preview.route_explanation,
                "session_plan": preview.session_plan,
                "guardrails": preview.guardrails,
                "budget": preview.budget,
                "context_plan": preview.context_plan,
                "optimization": preview.optimization,
                "options": preview.options,
            }
            latency_ms = (time.perf_counter() - start) * 1000.0
            manager.trace_store.add(
                RequestTrace(
                    timestamp_ms=int(time.time() * 1000),
                    endpoint="/session/{session_id}/dry-run",
                    session_id=session_id,
                    provider=preview.provider,
                    model=preview.model,
                    route_reason=preview.route_reason,
                    status_code=200,
                    latency_ms=latency_ms,
                    budget_action=str((preview.budget or {}).get("action", "")),
                    budget_status=str((preview.budget or {}).get("status", "")),
                    budget_reason=str((preview.budget or {}).get("reason", "")),
                    usage={},
                    extra={"optimization": preview.optimization},
                )
            )
            manager.metrics_collector.record(latency_ms=latency_ms, status_code=200, is_dry_run=True)
            return payload

        if manager.wrapper is not None:
            payload = {
                "dry_run": True,
                "provider": "wrapper",
                "model": getattr(manager.wrapper, "llm_model", "unknown"),
                "route_reason": "wrapper-fallback",
                "budget": {},
                "context_plan": {},
                "optimization": {},
                "options": merged_options,
            }
            latency_ms = (time.perf_counter() - start) * 1000.0
            manager.trace_store.add(
                RequestTrace(
                    timestamp_ms=int(time.time() * 1000),
                    endpoint="/session/{session_id}/dry-run",
                    session_id=session_id,
                    provider="wrapper",
                    model=getattr(manager.wrapper, "llm_model", "unknown"),
                    route_reason="wrapper-fallback",
                    status_code=200,
                    latency_ms=latency_ms,
                    usage={},
                )
            )
            manager.metrics_collector.record(latency_ms=latency_ms, status_code=200, is_dry_run=True)
            return payload

        raise HTTPException(status_code=500, detail="No orchestrator or wrapper available")

    @app.post("/session/{session_id}/stream")
    async def stream_message(session_id: str, req: StreamRequest):
        start = time.perf_counter()
        session_lock = manager.get_or_create_session_lock(session_id)
        async with session_lock:
            try:
                sess = await manager.get_session(session_id)
            except KeyError:
                raise HTTPException(status_code=404, detail="session not found")

            merged_options = manager._with_session_options(req.options, sess)
            rate_limit = manager.rate_limiter.allow(
                session_id=session_id,
                endpoint="/session/{session_id}/stream",
                options=merged_options,
            )
            if not rate_limit.allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "message": "Rate limit exceeded",
                        "retry_after_ms": rate_limit.retry_after_ms,
                        "reason": rate_limit.reason,
                    },
                )

            if manager.orchestrator is None:
                raise HTTPException(status_code=501, detail="Streaming requires an orchestrator")

            try:
                meta, chunk_iter = await manager.orchestrator.run_query_stream_async(
                    QueryRequest(
                        user_query=req.message,
                        session_id=session_id,
                        system_prompt=sess.system_prompt,
                        metadata_filter=sess.metadata,
                        options=merged_options,
                    )
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            sess.chat_history.append({"role": "user", "content": req.message})

        async def _stream() -> Any:
            chunks: list[str] = []
            stream_status = 200
            stream_error = ""
            try:
                async for chunk in chunk_iter:
                    chunks.append(chunk)
                    yield chunk
            except Exception as e:
                stream_status = 500
                stream_error = str(e)
                raise
            finally:
                reply = "".join(chunks)
                async with session_lock:
                    if stream_status == 200:
                        sess.chat_history.append({"role": "assistant", "content": reply})
                latency_ms = (time.perf_counter() - start) * 1000.0
                manager.trace_store.add(
                    RequestTrace(
                        timestamp_ms=int(time.time() * 1000),
                        endpoint="/session/{session_id}/stream",
                        session_id=session_id,
                        provider=meta.provider,
                        model=meta.model,
                        route_reason=meta.route_reason,
                        status_code=stream_status,
                        latency_ms=latency_ms,
                        budget_action=str(meta.budget_decision.get("action", "")),
                        budget_status=str(meta.budget_decision.get("status", "")),
                        budget_reason=str(meta.budget_decision.get("reason", "")),
                        usage=meta.usage,
                        extra={"stream_error": stream_error} if stream_error else {},
                    )
                )
                manager.metrics_collector.record(latency_ms=latency_ms, status_code=stream_status)

        return StreamingResponse(_stream(), media_type="text/plain")

    @app.get("/session/{session_id}")
    async def session_state(session_id: str):
        try:
            sess = await manager.get_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found")
        return {
            "session_id": sess.session_id,
            "system_prompt": sess.system_prompt,
            "metadata": sess.metadata,
            "chat_history": sess.chat_history,
            "running_summary": sess.running_summary,
        }

    @app.delete("/session/{session_id}")
    async def close(session_id: str):
        await manager.close_session(session_id)
        return {"closed": True}

    @app.get("/debug/traces")
    async def debug_traces(limit: int = 20):
        traces = manager.trace_store.latest(limit=limit)
        return {
            "count": len(traces),
            "traces": [
                {
                    "timestamp_ms": t.timestamp_ms,
                    "endpoint": t.endpoint,
                    "session_id": t.session_id,
                    "provider": t.provider,
                    "model": t.model,
                    "route_reason": t.route_reason,
                    "status_code": t.status_code,
                    "latency_ms": t.latency_ms,
                    "budget_action": t.budget_action,
                    "budget_status": t.budget_status,
                    "budget_reason": t.budget_reason,
                    "usage": t.usage,
                    "extra": t.extra,
                }
                for t in traces
            ],
        }

    @app.get("/debug/metrics")
    async def debug_metrics():
        snapshot = manager.metrics_collector.snapshot()
        return {
            "request_count": snapshot.request_count,
            "error_count": snapshot.error_count,
            "dry_run_count": snapshot.dry_run_count,
            "avg_latency_ms": snapshot.avg_latency_ms,
            "avg_input_tokens": snapshot.avg_input_tokens,
            "avg_output_tokens": snapshot.avg_output_tokens,
            "avg_total_tokens": snapshot.avg_total_tokens,
        }

    return app
