from __future__ import annotations

import math

from ollama_wrapper.orchestration.contracts import QueryRequest
from ollama_wrapper.orchestration.policies import BudgetDecision, BudgetPolicy


class TokenBudgetPolicy(BudgetPolicy):
    """Enforce per-request input token budgets with configurable behavior modes."""

    def __init__(self, default_max_input_tokens: int = 1200, default_mode: str = "warn") -> None:
        self.default_max_input_tokens = default_max_input_tokens
        self.default_mode = default_mode

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(math.ceil(len(text.split()) * 1.3)))

    @staticmethod
    def _truncate_to_budget(text: str, max_input_tokens: int) -> str:
        if not text:
            return text
        max_words = max(1, int(max_input_tokens / 1.3))
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    def enforce_budget(self, request: QueryRequest) -> BudgetDecision:
        options = request.options or {}
        max_input_tokens = int(options.get("max_input_tokens", self.default_max_input_tokens))
        max_output_tokens = options.get("max_output_tokens")
        mode = str(options.get("budget_mode", self.default_mode)).lower()
        if mode not in {"warn", "truncate", "reject"}:
            mode = self.default_mode

        estimated = self._estimate_tokens(request.user_query)
        if estimated <= max_input_tokens:
            return BudgetDecision(
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                mode=mode,
                status="within-budget",
                action="allow",
                estimated_input_tokens=estimated,
                effective_input_tokens=estimated,
                reason="input-within-limit",
            )

        if mode == "warn":
            return BudgetDecision(
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                mode=mode,
                status="over-budget",
                action="allow",
                estimated_input_tokens=estimated,
                effective_input_tokens=estimated,
                reason="over-limit-warning",
            )

        if mode == "truncate":
            adjusted = self._truncate_to_budget(request.user_query, max_input_tokens)
            return BudgetDecision(
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                mode=mode,
                status="over-budget",
                action="truncate",
                estimated_input_tokens=estimated,
                effective_input_tokens=self._estimate_tokens(adjusted),
                adjusted_user_query=adjusted,
                reason="over-limit-truncated",
            )

        return BudgetDecision(
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            mode=mode,
            status="over-budget",
            action="reject",
            estimated_input_tokens=estimated,
            effective_input_tokens=max_input_tokens,
            reason="over-limit-rejected",
        )
