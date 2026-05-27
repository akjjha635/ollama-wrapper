from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RequestTrace:
    timestamp_ms: int
    endpoint: str
    session_id: str | None
    provider: str
    model: str
    route_reason: str
    status_code: int
    latency_ms: float
    budget_action: str = ""
    budget_status: str = ""
    budget_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricsSnapshot:
    request_count: int
    error_count: int
    dry_run_count: int
    avg_latency_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
