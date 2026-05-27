from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BenchmarkSample:
    latency_ms: float
    status_code: int
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class BenchmarkReport:
    request_count: int
    success_count: int
    error_count: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    samples: list[BenchmarkSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["samples"] = [asdict(s) for s in self.samples]
        return payload


@dataclass(slots=True)
class ComparativeBenchmarkReport:
    baseline_label: str
    candidate_label: str
    baseline: BenchmarkReport
    candidate: BenchmarkReport
    deltas: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "deltas": self.deltas,
        }


class BenchmarkRunner:
    """Run simple latency/token benchmark against API message endpoint."""

    def __init__(self, base_url: str, timeout_sec: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def _request_json(self, path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url=f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
                return int(resp.status), json.loads(body) if body else {}
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8")
            try:
                parsed = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {"detail": body}
            return int(err.code), parsed

    @staticmethod
    def summarize(samples: list[BenchmarkSample]) -> BenchmarkReport:
        if not samples:
            return BenchmarkReport(
                request_count=0,
                success_count=0,
                error_count=0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                avg_input_tokens=0.0,
                avg_output_tokens=0.0,
                avg_total_tokens=0.0,
                samples=[],
            )

        latencies = sorted(s.latency_ms for s in samples)
        total = len(samples)
        success_count = sum(1 for s in samples if s.status_code < 400)
        error_count = total - success_count

        p50 = statistics.median(latencies)
        p95_index = min(total - 1, max(0, int(round(0.95 * total)) - 1))
        p95 = latencies[p95_index]

        return BenchmarkReport(
            request_count=total,
            success_count=success_count,
            error_count=error_count,
            avg_latency_ms=float(sum(latencies) / total),
            p50_latency_ms=float(p50),
            p95_latency_ms=float(p95),
            avg_input_tokens=float(sum(s.input_tokens for s in samples) / total),
            avg_output_tokens=float(sum(s.output_tokens for s in samples) / total),
            avg_total_tokens=float(sum(s.total_tokens for s in samples) / total),
            samples=samples,
        )

    @staticmethod
    def compare_reports(
        baseline: BenchmarkReport,
        candidate: BenchmarkReport,
        *,
        baseline_label: str = "baseline",
        candidate_label: str = "candidate",
    ) -> ComparativeBenchmarkReport:
        deltas = {
            "avg_latency_ms": candidate.avg_latency_ms - baseline.avg_latency_ms,
            "p50_latency_ms": candidate.p50_latency_ms - baseline.p50_latency_ms,
            "p95_latency_ms": candidate.p95_latency_ms - baseline.p95_latency_ms,
            "avg_input_tokens": candidate.avg_input_tokens - baseline.avg_input_tokens,
            "avg_output_tokens": candidate.avg_output_tokens - baseline.avg_output_tokens,
            "avg_total_tokens": candidate.avg_total_tokens - baseline.avg_total_tokens,
            "error_count": float(candidate.error_count - baseline.error_count),
            "success_count": float(candidate.success_count - baseline.success_count),
        }
        return ComparativeBenchmarkReport(
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            baseline=baseline,
            candidate=candidate,
            deltas=deltas,
        )

    def run_message_benchmark(
        self,
        *,
        iterations: int = 20,
        message: str = "hello",
        system_prompt: str = "You are helpful.",
        options: dict[str, Any] | None = None,
    ) -> BenchmarkReport:
        status, created = self._request_json("/session", method="POST", payload={"system_prompt": system_prompt})
        if status >= 400:
            return BenchmarkReport(
                request_count=1,
                success_count=0,
                error_count=1,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                avg_input_tokens=0.0,
                avg_output_tokens=0.0,
                avg_total_tokens=0.0,
                samples=[BenchmarkSample(latency_ms=0.0, status_code=status)],
            )

        session_id = created["session_id"]
        samples: list[BenchmarkSample] = []

        try:
            for _ in range(max(1, int(iterations))):
                start = time.perf_counter()
                status_code, payload = self._request_json(
                    f"/session/{session_id}/message",
                    method="POST",
                    payload={"message": message, "options": options or {}},
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
                samples.append(
                    BenchmarkSample(
                        latency_ms=elapsed_ms,
                        status_code=status_code,
                        input_tokens=int(usage.get("input_tokens") or 0),
                        output_tokens=int(usage.get("output_tokens") or 0),
                        total_tokens=int(usage.get("total_tokens") or 0),
                    )
                )
        finally:
            self._request_json(f"/session/{session_id}", method="DELETE")

        return self.summarize(samples)
