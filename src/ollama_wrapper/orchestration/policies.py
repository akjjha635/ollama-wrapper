from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import QueryRequest


@dataclass(slots=True)
class RoutingDecision:
    provider: str
    model: str
    reason: str = ""


@dataclass(slots=True)
class BudgetDecision:
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    mode: str = "warn"
    status: str = "not-enforced"
    action: str = "allow"
    estimated_input_tokens: int | None = None
    effective_input_tokens: int | None = None
    adjusted_user_query: str | None = None
    reason: str = ""
    cost_guardrails: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextPlan:
    max_context_items: int = 8
    include_summary: bool = True
    include_metadata_filter: bool = True


@dataclass(slots=True)
class SessionPlan:
    query_type: str = "general"
    session_turn_count: int = 0
    include_summary: bool = False
    planning_reason: str = "default"


@dataclass(slots=True)
class GuardrailsDecision:
    status: str = "not-enforced"
    action: str = "allow"
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class RoutingPolicy(Protocol):
    def choose_route(self, request: QueryRequest) -> RoutingDecision:
        ...


class BudgetPolicy(Protocol):
    def enforce_budget(self, request: QueryRequest) -> BudgetDecision:
        ...


class ContextPolicy(Protocol):
    def build_context_plan(self, request: QueryRequest) -> ContextPlan:
        ...


class QueryPlanner(Protocol):
    def build_plan(self, request: QueryRequest) -> SessionPlan:
        ...


class GuardrailsPolicy(Protocol):
    def evaluate(self, request: QueryRequest, route: RoutingDecision) -> GuardrailsDecision:
        ...


class StaticRoutingPolicy:
    """Simple default routing policy for incremental adoption."""

    def __init__(self, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model

    def choose_route(self, request: QueryRequest) -> RoutingDecision:
        return RoutingDecision(provider=self._provider, model=self._model, reason="static-route")


class HeuristicRoutingPolicy:
    """Routes requests by simple complexity and budget hints.

    This policy is intentionally lightweight so teams can start with deterministic
    behavior and later replace it with learned/telemetry-driven routing.
    """

    def __init__(
        self,
        default_provider: str,
        default_model: str,
        complex_model: str | None = None,
        complexity_word_threshold: int = 40,
    ) -> None:
        self.default_provider = default_provider
        self.default_model = default_model
        self.complex_model = complex_model or default_model
        self.complexity_word_threshold = complexity_word_threshold

    def choose_route(self, request: QueryRequest) -> RoutingDecision:
        options = request.options or {}

        preferred_provider = options.get("preferred_provider")
        preferred_model = options.get("preferred_model")
        if preferred_provider or preferred_model:
            return RoutingDecision(
                provider=preferred_provider or self.default_provider,
                model=preferred_model or self.default_model,
                reason="explicit-preference",
            )

        budget_tier = str(options.get("budget_tier", "")).lower()
        if budget_tier == "low":
            cheap_model = options.get("cheap_model") or self.default_model
            return RoutingDecision(provider=self.default_provider, model=cheap_model, reason="budget-low")

        word_count = len((request.user_query or "").split())
        if word_count >= self.complexity_word_threshold:
            return RoutingDecision(
                provider=self.default_provider,
                model=self.complex_model,
                reason="complexity-high",
            )

        return RoutingDecision(provider=self.default_provider, model=self.default_model, reason="complexity-low")


class NoopBudgetPolicy:
    def enforce_budget(self, request: QueryRequest) -> BudgetDecision:
        return BudgetDecision()


class DefaultContextPolicy:
    def build_context_plan(self, request: QueryRequest) -> ContextPlan:
        return ContextPlan(
            max_context_items=8,
            include_summary=bool(request.session_id),
            include_metadata_filter=bool(request.metadata_filter),
        )


class SessionAwareQueryPlanner:
    """Lightweight planner that infers query type and session hints."""

    @staticmethod
    def _infer_query_type(request: QueryRequest) -> tuple[str, str]:
        query = (request.user_query or "").strip().lower()
        options = request.options or {}
        explicit = str(options.get("query_type", "")).strip().lower()
        if explicit:
            return explicit, "explicit-query-type"

        if any(token in query for token in ["why", "explain", "tradeoff", "design"]):
            return "reasoning", "reasoning-keywords"
        if any(token in query for token in ["where", "when", "who", "what", "how many"]):
            return "factoid", "factoid-keywords"
        if any(token in query for token in ["code", "function", "class", "bug", "test"]):
            return "code", "code-keywords"
        return "general", "default-general"

    def build_plan(self, request: QueryRequest) -> SessionPlan:
        options = request.options or {}
        query_type, reason = self._infer_query_type(request)
        turn_count = int(options.get("session_turn_count", 0) or 0)
        include_summary = bool(request.session_id) or turn_count >= 2 or query_type in {"reasoning", "code"}
        return SessionPlan(
            query_type=query_type,
            session_turn_count=max(0, turn_count),
            include_summary=include_summary,
            planning_reason=reason,
        )


class NoopGuardrailsPolicy:
    def evaluate(self, request: QueryRequest, route: RoutingDecision) -> GuardrailsDecision:
        _ = request
        _ = route
        return GuardrailsDecision(status="not-enforced", action="allow", reason="guardrails-disabled", details={})
