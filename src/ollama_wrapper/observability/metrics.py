from __future__ import annotations

from .contracts import MetricsSnapshot


class MetricsCollector:
    """Simple aggregate metrics collector for request-level telemetry."""

    def __init__(self) -> None:
        self.request_count = 0
        self.error_count = 0
        self.dry_run_count = 0
        self.total_latency_ms = 0.0
        self.total_input_tokens = 0.0
        self.total_output_tokens = 0.0
        self.total_tokens = 0.0

    def record(
        self,
        *,
        latency_ms: float,
        status_code: int,
        input_tokens: float | int | None = None,
        output_tokens: float | int | None = None,
        total_tokens: float | int | None = None,
        is_dry_run: bool = False,
    ) -> None:
        self.request_count += 1
        self.total_latency_ms += float(latency_ms)
        if status_code >= 400:
            self.error_count += 1
        if is_dry_run:
            self.dry_run_count += 1

        self.total_input_tokens += float(input_tokens or 0.0)
        self.total_output_tokens += float(output_tokens or 0.0)
        self.total_tokens += float(total_tokens or 0.0)

    def snapshot(self) -> MetricsSnapshot:
        req = max(1, self.request_count)
        return MetricsSnapshot(
            request_count=self.request_count,
            error_count=self.error_count,
            dry_run_count=self.dry_run_count,
            avg_latency_ms=self.total_latency_ms / req,
            avg_input_tokens=self.total_input_tokens / req,
            avg_output_tokens=self.total_output_tokens / req,
            avg_total_tokens=self.total_tokens / req,
        )
