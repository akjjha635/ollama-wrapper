from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ollama_wrapper.orchestration.contracts import QueryRequest
from ollama_wrapper.orchestration.policies import GuardrailsDecision, GuardrailsPolicy, RoutingDecision


@dataclass(slots=True)
class GovernanceConfig:
    allowed_providers: set[str] = field(default_factory=set)
    allowed_models: set[str] = field(default_factory=set)
    max_payload_chars: int = 12000
    tenant_overrides: dict[str, "TenantGovernanceRule"] = field(default_factory=dict)


@dataclass(slots=True)
class TenantGovernanceRule:
    allowed_providers: set[str] = field(default_factory=set)
    allowed_models: set[str] = field(default_factory=set)
    max_payload_chars: int | None = None


class GovernancePolicy(GuardrailsPolicy):
    """Validate route and payload constraints before provider execution."""

    def __init__(self, config: GovernanceConfig | None = None) -> None:
        self.config = config or GovernanceConfig()

    def _tenant_id(self, request: QueryRequest) -> str | None:
        options = request.options or {}
        tenant_opt = options.get("tenant_id")
        if isinstance(tenant_opt, str) and tenant_opt.strip():
            return tenant_opt.strip()
        metadata = request.metadata_filter or {}
        tenant_meta = metadata.get("tenant_id")
        if isinstance(tenant_meta, str) and tenant_meta.strip():
            return tenant_meta.strip()
        return None

    def _effective_rules(self, request: QueryRequest) -> tuple[set[str], set[str], int]:
        base_allowed_providers = set(self.config.allowed_providers)
        base_allowed_models = set(self.config.allowed_models)
        base_max_payload = int(self.config.max_payload_chars)

        tenant_id = self._tenant_id(request)
        overrides = getattr(self.config, "tenant_overrides", {})
        if not tenant_id or tenant_id not in overrides:
            return base_allowed_providers, base_allowed_models, base_max_payload

        tenant_rule = overrides[tenant_id]
        providers = set(tenant_rule.allowed_providers) if tenant_rule.allowed_providers else base_allowed_providers
        models = set(tenant_rule.allowed_models) if tenant_rule.allowed_models else base_allowed_models
        max_payload = int(tenant_rule.max_payload_chars) if tenant_rule.max_payload_chars is not None else base_max_payload
        return providers, models, max_payload

    def evaluate(self, request: QueryRequest, route: RoutingDecision) -> GuardrailsDecision:
        allowed_providers, allowed_models, max_payload_chars = self._effective_rules(request)
        tenant_id = self._tenant_id(request)
        details: dict[str, Any] = {
            "provider": route.provider,
            "model": route.model,
            "query_chars": len(request.user_query or ""),
            "max_payload_chars": max_payload_chars,
            "tenant_id": tenant_id,
        }

        if allowed_providers and route.provider not in allowed_providers:
            return GuardrailsDecision(
                status="rejected",
                action="reject",
                reason="provider-not-allowed",
                details=details,
            )

        if allowed_models and route.model not in allowed_models:
            return GuardrailsDecision(
                status="rejected",
                action="reject",
                reason="model-not-allowed",
                details=details,
            )

        if len(request.user_query or "") > max(1, int(max_payload_chars)):
            return GuardrailsDecision(
                status="rejected",
                action="reject",
                reason="payload-too-large",
                details=details,
            )

        return GuardrailsDecision(
            status="passed",
            action="allow",
            reason="governance-passed",
            details=details,
        )
