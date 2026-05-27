from __future__ import annotations

from typing import Any, AsyncIterator

from ollama_wrapper.llm import LLMMessage, LLMProvider
from ollama_wrapper.optimization import MathematicalOptimizationLayer, OptimizationConfig

from .contracts import QueryOrchestrator, QueryRequest, QueryResponse, RoutePreview
from .policies import (
    BudgetPolicy,
    ContextPolicy,
    DefaultContextPolicy,
    GuardrailsPolicy,
    NoopBudgetPolicy,
    NoopGuardrailsPolicy,
    QueryPlanner,
    RoutingPolicy,
    SessionAwareQueryPlanner,
    StaticRoutingPolicy,
)


class DefaultQueryOrchestrator(QueryOrchestrator):
    """Policy-driven query orchestration over pluggable LLM providers."""

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        routing_policy: RoutingPolicy,
        budget_policy: BudgetPolicy | None = None,
        context_policy: ContextPolicy | None = None,
        query_planner: QueryPlanner | None = None,
        guardrails_policy: GuardrailsPolicy | None = None,
        optimization_layer: MathematicalOptimizationLayer | None = None,
    ) -> None:
        self.providers = providers
        self.routing_policy = routing_policy
        self.budget_policy = budget_policy or NoopBudgetPolicy()
        self.context_policy = context_policy or DefaultContextPolicy()
        self.query_planner = query_planner or SessionAwareQueryPlanner()
        self.guardrails_policy = guardrails_policy or NoopGuardrailsPolicy()
        self.optimization_layer = optimization_layer or MathematicalOptimizationLayer()

    @classmethod
    def from_single_provider(
        cls,
        provider_key: str,
        provider: LLMProvider,
        model: str,
    ) -> "DefaultQueryOrchestrator":
        return cls(
            providers={provider_key: provider},
            routing_policy=StaticRoutingPolicy(provider=provider_key, model=model),
        )

    def _build_messages(self, request: QueryRequest, _context_plan: Any) -> list[LLMMessage]:
        messages: list[LLMMessage] = []
        if request.system_prompt:
            messages.append(LLMMessage(role="system", content=request.system_prompt))

        raw_candidates = request.options.get("rag_candidates", []) if request.options else []
        candidates = self.optimization_layer.parse_candidates(raw_candidates)
        if candidates:
            token_budget = int(request.options.get("token_budget", 1200)) if request.options else 1200
            semantic_weight = float(request.options.get("semantic_weight", 0.7)) if request.options else 0.7
            lexical_weight = float(request.options.get("lexical_weight", 0.3)) if request.options else 0.3
            max_items = int(getattr(_context_plan, "max_context_items", 8))
            config = OptimizationConfig(
                semantic_weight=semantic_weight,
                lexical_weight=lexical_weight,
                token_budget=token_budget,
                max_context_items=max_items,
                query_type=str(request.options.get("query_type", "general")) if request.options else "general",
            )
            optimization = self.optimization_layer.optimize_context(candidates, config)
            if optimization.selected_context:
                context_blob = "\n\n".join(optimization.selected_context)
                messages.append(
                    LLMMessage(
                        role="system",
                        content=(
                            "### OPTIMIZED RAG CONTEXT\n"
                            f"{context_blob}"
                        ),
                    )
                )

        messages.append(LLMMessage(role="user", content=request.user_query))
        return messages

    @staticmethod
    def _budget_dict(budget: Any) -> dict[str, Any]:
        return {
            "max_input_tokens": budget.max_input_tokens,
            "max_output_tokens": budget.max_output_tokens,
            "mode": budget.mode,
            "status": budget.status,
            "action": budget.action,
            "estimated_input_tokens": budget.estimated_input_tokens,
            "effective_input_tokens": budget.effective_input_tokens,
            "reason": budget.reason,
            "cost_guardrails": budget.cost_guardrails,
        }

    @staticmethod
    def _route_explanation(route: Any, session_plan: Any) -> dict[str, Any]:
        return {
            "provider": route.provider,
            "model": route.model,
            "reason": route.reason,
            "strategy": "policy-routing-v1",
            "query_type": getattr(session_plan, "query_type", "general"),
            "planning_reason": getattr(session_plan, "planning_reason", "default"),
        }

    @staticmethod
    def _apply_budget(request: QueryRequest, budget: Any) -> QueryRequest:
        if budget.action == "reject":
            raise ValueError(
                f"Request exceeds input token budget ({budget.estimated_input_tokens} > {budget.max_input_tokens})."
            )
        if budget.action == "truncate" and budget.adjusted_user_query is not None:
            return QueryRequest(
                user_query=budget.adjusted_user_query,
                session_id=request.session_id,
                system_prompt=request.system_prompt,
                metadata_filter=request.metadata_filter,
                response_schema=request.response_schema,
                options=request.options,
            )
        return request

    @staticmethod
    def _apply_budget_preview(request: QueryRequest, budget: Any) -> QueryRequest:
        if budget.action == "truncate" and budget.adjusted_user_query is not None:
            return QueryRequest(
                user_query=budget.adjusted_user_query,
                session_id=request.session_id,
                system_prompt=request.system_prompt,
                metadata_filter=request.metadata_filter,
                response_schema=request.response_schema,
                options=request.options,
            )
        return request

    def preview_route(self, request: QueryRequest) -> RoutePreview:
        session_plan = self.query_planner.build_plan(request)
        route = self.routing_policy.choose_route(request)
        guardrails = self.guardrails_policy.evaluate(request, route)
        budget = self.budget_policy.enforce_budget(request)
        effective_request = self._apply_budget_preview(request, budget)
        context_plan = self.context_policy.build_context_plan(request)
        raw_candidates = effective_request.options.get("rag_candidates", []) if effective_request.options else []
        candidates = self.optimization_layer.parse_candidates(raw_candidates)
        token_budget = int(effective_request.options.get("token_budget", 1200)) if effective_request.options else 1200
        semantic_weight = float(effective_request.options.get("semantic_weight", 0.7)) if effective_request.options else 0.7
        lexical_weight = float(effective_request.options.get("lexical_weight", 0.3)) if effective_request.options else 0.3
        config = OptimizationConfig(
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            token_budget=token_budget,
            max_context_items=context_plan.max_context_items,
            query_type=session_plan.query_type,
        )
        optimization = self.optimization_layer.optimize_context(candidates, config)
        optimization_summary = self.optimization_layer.summarize(candidates, optimization, config)
        merged_options = {
            **effective_request.options,
            "query_type": session_plan.query_type,
            "max_input_tokens": budget.max_input_tokens,
            "max_output_tokens": budget.max_output_tokens,
            "cost_guardrails": budget.cost_guardrails,
            "context_plan": {
                "max_context_items": context_plan.max_context_items,
                "include_summary": context_plan.include_summary,
                "include_metadata_filter": context_plan.include_metadata_filter,
            },
            "optimization": {
                "candidate_count": optimization_summary.candidate_count,
                "selected_count": optimization_summary.selected_count,
                "token_estimate": optimization_summary.token_estimate,
                "token_budget": optimization_summary.token_budget,
                "score_weights": optimization_summary.score_weights,
                "selected_indices": optimization.selected_indices,
                "confidence_scores": optimization.confidence_scores,
                "avg_confidence": optimization_summary.avg_confidence,
                "max_confidence": optimization_summary.max_confidence,
            },
            "budget_decision": self._budget_dict(budget),
            "guardrails": {
                "status": guardrails.status,
                "action": guardrails.action,
                "reason": guardrails.reason,
                "details": guardrails.details,
            },
            "session_plan": {
                "query_type": session_plan.query_type,
                "session_turn_count": session_plan.session_turn_count,
                "include_summary": session_plan.include_summary,
                "planning_reason": session_plan.planning_reason,
            },
        }
        return RoutePreview(
            provider=route.provider,
            model=route.model,
            route_reason=route.reason,
            route_explanation=self._route_explanation(route, session_plan),
            session_plan=merged_options["session_plan"],
            guardrails=merged_options["guardrails"],
            budget=self._budget_dict(budget),
            context_plan={
                "max_context_items": context_plan.max_context_items,
                "include_summary": context_plan.include_summary,
                "include_metadata_filter": context_plan.include_metadata_filter,
            },
            optimization=merged_options["optimization"],
            options=merged_options,
        )

    def run_query(self, request: QueryRequest) -> QueryResponse:
        session_plan = self.query_planner.build_plan(request)
        route = self.routing_policy.choose_route(request)
        provider = self.providers[route.provider]
        guardrails = self.guardrails_policy.evaluate(request, route)
        if guardrails.action == "reject":
            raise ValueError(f"Guardrails rejected request: {guardrails.reason}")
        budget = self.budget_policy.enforce_budget(request)
        effective_request = self._apply_budget(request, budget)
        context_plan = self.context_policy.build_context_plan(request)

        merged_options = {
            **effective_request.options,
            "query_type": session_plan.query_type,
            "max_input_tokens": budget.max_input_tokens,
            "max_output_tokens": budget.max_output_tokens,
            "cost_guardrails": budget.cost_guardrails,
            "budget_decision": self._budget_dict(budget),
            "guardrails": {
                "status": guardrails.status,
                "action": guardrails.action,
                "reason": guardrails.reason,
                "details": guardrails.details,
            },
            "session_plan": {
                "query_type": session_plan.query_type,
                "session_turn_count": session_plan.session_turn_count,
                "include_summary": session_plan.include_summary,
                "planning_reason": session_plan.planning_reason,
            },
            "context_plan": {
                "max_context_items": context_plan.max_context_items,
                "include_summary": context_plan.include_summary,
                "include_metadata_filter": context_plan.include_metadata_filter,
            },
        }

        result = provider.chat(
            model=route.model,
            messages=self._build_messages(effective_request, context_plan),
            response_schema=effective_request.response_schema,
            options=merged_options,
        )
        return QueryResponse.from_chat_result(
            provider=provider.provider_name(),
            result=result,
            route_reason=route.reason,
            route_explanation=self._route_explanation(route, session_plan),
            budget_decision=self._budget_dict(budget),
            guardrails=merged_options["guardrails"],
            session_plan=merged_options["session_plan"],
        )

    async def run_query_async(self, request: QueryRequest) -> QueryResponse:
        session_plan = self.query_planner.build_plan(request)
        route = self.routing_policy.choose_route(request)
        provider = self.providers[route.provider]
        guardrails = self.guardrails_policy.evaluate(request, route)
        if guardrails.action == "reject":
            raise ValueError(f"Guardrails rejected request: {guardrails.reason}")
        budget = self.budget_policy.enforce_budget(request)
        effective_request = self._apply_budget(request, budget)
        context_plan = self.context_policy.build_context_plan(request)

        merged_options = {
            **effective_request.options,
            "query_type": session_plan.query_type,
            "max_input_tokens": budget.max_input_tokens,
            "max_output_tokens": budget.max_output_tokens,
            "cost_guardrails": budget.cost_guardrails,
            "budget_decision": self._budget_dict(budget),
            "guardrails": {
                "status": guardrails.status,
                "action": guardrails.action,
                "reason": guardrails.reason,
                "details": guardrails.details,
            },
            "session_plan": {
                "query_type": session_plan.query_type,
                "session_turn_count": session_plan.session_turn_count,
                "include_summary": session_plan.include_summary,
                "planning_reason": session_plan.planning_reason,
            },
            "context_plan": {
                "max_context_items": context_plan.max_context_items,
                "include_summary": context_plan.include_summary,
                "include_metadata_filter": context_plan.include_metadata_filter,
            },
        }

        result = await provider.chat_async(
            model=route.model,
            messages=self._build_messages(effective_request, context_plan),
            response_schema=effective_request.response_schema,
            options=merged_options,
        )
        return QueryResponse.from_chat_result(
            provider=provider.provider_name(),
            result=result,
            route_reason=route.reason,
            route_explanation=self._route_explanation(route, session_plan),
            budget_decision=self._budget_dict(budget),
            guardrails=merged_options["guardrails"],
            session_plan=merged_options["session_plan"],
        )

    async def run_query_stream_async(self, request: QueryRequest) -> tuple[QueryResponse, AsyncIterator[str]]:
        session_plan = self.query_planner.build_plan(request)
        route = self.routing_policy.choose_route(request)
        provider = self.providers[route.provider]
        guardrails = self.guardrails_policy.evaluate(request, route)
        if guardrails.action == "reject":
            raise ValueError(f"Guardrails rejected request: {guardrails.reason}")
        budget = self.budget_policy.enforce_budget(request)
        effective_request = self._apply_budget(request, budget)
        context_plan = self.context_policy.build_context_plan(request)

        merged_options = {
            **effective_request.options,
            "query_type": session_plan.query_type,
            "max_input_tokens": budget.max_input_tokens,
            "max_output_tokens": budget.max_output_tokens,
            "cost_guardrails": budget.cost_guardrails,
            "budget_decision": self._budget_dict(budget),
            "guardrails": {
                "status": guardrails.status,
                "action": guardrails.action,
                "reason": guardrails.reason,
                "details": guardrails.details,
            },
            "session_plan": {
                "query_type": session_plan.query_type,
                "session_turn_count": session_plan.session_turn_count,
                "include_summary": session_plan.include_summary,
                "planning_reason": session_plan.planning_reason,
            },
            "context_plan": {
                "max_context_items": context_plan.max_context_items,
                "include_summary": context_plan.include_summary,
                "include_metadata_filter": context_plan.include_metadata_filter,
            },
        }

        meta_response = QueryResponse(
            reply="",
            provider=provider.provider_name(),
            model=route.model,
            route_reason=route.reason,
            route_explanation=self._route_explanation(route, session_plan),
            usage={},
            budget_decision=self._budget_dict(budget),
            guardrails=merged_options["guardrails"],
            session_plan=merged_options["session_plan"],
            raw={},
        )

        supports_stream = provider.__class__.chat_stream_async is not LLMProvider.chat_stream_async
        if supports_stream:
            async def _stream() -> AsyncIterator[str]:
                async for chunk in provider.chat_stream_async(
                    model=route.model,
                    messages=self._build_messages(effective_request, context_plan),
                    response_schema=effective_request.response_schema,
                    options=merged_options,
                ):
                    yield chunk

            return meta_response, _stream()

        result = await provider.chat_async(
            model=route.model,
            messages=self._build_messages(effective_request, context_plan),
            response_schema=effective_request.response_schema,
            options=merged_options,
        )
        full_response = QueryResponse.from_chat_result(
            provider=provider.provider_name(),
            result=result,
            route_reason=route.reason,
            route_explanation=self._route_explanation(route, session_plan),
            budget_decision=self._budget_dict(budget),
            guardrails=merged_options["guardrails"],
            session_plan=merged_options["session_plan"],
        )

        async def _single_chunk() -> AsyncIterator[str]:
            yield result.content

        return full_response, _single_chunk()
