from __future__ import annotations

from typing import Any


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def comparison_to_markdown(report: dict[str, Any]) -> str:
    baseline_label = str(report.get("baseline_label", "baseline"))
    candidate_label = str(report.get("candidate_label", "candidate"))
    baseline = report.get("baseline", {}) or {}
    candidate = report.get("candidate", {}) or {}
    deltas = report.get("deltas", {}) or {}

    metrics = [
        ("request_count", baseline.get("request_count", 0), candidate.get("request_count", 0), deltas.get("request_count", 0.0)),
        ("success_count", baseline.get("success_count", 0), candidate.get("success_count", 0), deltas.get("success_count", 0.0)),
        ("error_count", baseline.get("error_count", 0), candidate.get("error_count", 0), deltas.get("error_count", 0.0)),
        ("avg_latency_ms", baseline.get("avg_latency_ms", 0.0), candidate.get("avg_latency_ms", 0.0), deltas.get("avg_latency_ms", 0.0)),
        ("p50_latency_ms", baseline.get("p50_latency_ms", 0.0), candidate.get("p50_latency_ms", 0.0), deltas.get("p50_latency_ms", 0.0)),
        ("p95_latency_ms", baseline.get("p95_latency_ms", 0.0), candidate.get("p95_latency_ms", 0.0), deltas.get("p95_latency_ms", 0.0)),
        ("avg_input_tokens", baseline.get("avg_input_tokens", 0.0), candidate.get("avg_input_tokens", 0.0), deltas.get("avg_input_tokens", 0.0)),
        ("avg_output_tokens", baseline.get("avg_output_tokens", 0.0), candidate.get("avg_output_tokens", 0.0), deltas.get("avg_output_tokens", 0.0)),
        ("avg_total_tokens", baseline.get("avg_total_tokens", 0.0), candidate.get("avg_total_tokens", 0.0), deltas.get("avg_total_tokens", 0.0)),
    ]

    lines = [
        "## Benchmark Comparison",
        "",
        f"Baseline: {baseline_label}",
        f"Candidate: {candidate_label}",
        "",
        f"| Metric | {baseline_label} | {candidate_label} | Delta ({candidate_label} - {baseline_label}) |",
        "|---|---:|---:|---:|",
    ]

    for metric, b_val, c_val, d_val in metrics:
        lines.append(f"| {metric} | {_fmt(b_val)} | {_fmt(c_val)} | {_fmt(d_val)} |")

    return "\n".join(lines)
