from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ollama_wrapper.llm import LLMChatResult


@dataclass(slots=True)
class QueryRequest:
    user_query: str
    session_id: str | None = None
    system_prompt: str = ""
    metadata_filter: dict[str, Any] = field(default_factory=dict)
    response_schema: dict[str, Any] | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryResponse:
    reply: str
    provider: str
    model: str
    route_reason: str = ""
    route_explanation: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    budget_decision: dict[str, Any] = field(default_factory=dict)
    guardrails: dict[str, Any] = field(default_factory=dict)
    session_plan: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chat_result(
        cls,
        provider: str,
        result: LLMChatResult,
        route_reason: str = "",
        route_explanation: dict[str, Any] | None = None,
        budget_decision: dict[str, Any] | None = None,
        guardrails: dict[str, Any] | None = None,
        session_plan: dict[str, Any] | None = None,
    ) -> "QueryResponse":
        return cls(
            reply=result.content,
            provider=provider,
            model=result.model,
            route_reason=route_reason,
            route_explanation=route_explanation or {},
            usage={
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
                "raw": result.usage.raw,
            },
            budget_decision=budget_decision or {},
            guardrails=guardrails or {},
            session_plan=session_plan or {},
            raw=result.raw,
        )


@dataclass(slots=True)
class RoutePreview:
    provider: str
    model: str
    route_reason: str = ""
    route_explanation: dict[str, Any] = field(default_factory=dict)
    session_plan: dict[str, Any] = field(default_factory=dict)
    guardrails: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    context_plan: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


class QueryOrchestrator(ABC):
    """Application-facing orchestration contract.

    Server-facing APIs can rely on this abstraction while the internals evolve.
    """

    @abstractmethod
    def run_query(self, request: QueryRequest) -> QueryResponse:
        raise NotImplementedError

    @abstractmethod
    async def run_query_async(self, request: QueryRequest) -> QueryResponse:
        raise NotImplementedError

    @abstractmethod
    def preview_route(self, request: QueryRequest) -> RoutePreview:
        """Return route and policy decisions without calling an LLM provider."""
        raise NotImplementedError

    async def run_query_stream_async(
        self,
        request: QueryRequest,
    ) -> tuple[QueryResponse, AsyncIterator[str]]:
        raise NotImplementedError("Streaming is not implemented for this orchestrator.")
