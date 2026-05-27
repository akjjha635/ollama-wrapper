#!/usr/bin/env python3
"""Run API message benchmark and print/save summary."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from ollama_wrapper.eval import BenchmarkRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a latency/token benchmark against the chat API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--candidate-base-url", default="", help="Optional candidate API URL for compare mode.")
    parser.add_argument("--compare", action="store_true", help="Run comparative benchmark (baseline vs candidate).")
    parser.add_argument("--baseline-label", default="linear")
    parser.add_argument("--candidate-label", default="faiss")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--message", default="hello")
    parser.add_argument("--system-prompt", default="You are helpful.")
    parser.add_argument("--options-json", default="{}", help="JSON string passed as message options.")
    parser.add_argument(
        "--candidate-options-json",
        default="",
        help="Optional JSON options for candidate run in compare mode. Defaults to --options-json.",
    )
    parser.add_argument("--output", default="", help="Optional JSON output file path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        options = json.loads(args.options_json)
    except json.JSONDecodeError as exc:
        print(f"Invalid --options-json: {exc}", file=sys.stderr)
        return 2

    candidate_options = options
    if args.candidate_options_json:
        try:
            candidate_options = json.loads(args.candidate_options_json)
        except json.JSONDecodeError as exc:
            print(f"Invalid --candidate-options-json: {exc}", file=sys.stderr)
            return 2

    runner = BenchmarkRunner(base_url=args.base_url)
    if args.compare:
        candidate_base_url = args.candidate_base_url or args.base_url
        candidate_runner = BenchmarkRunner(base_url=candidate_base_url)

        baseline_report = runner.run_message_benchmark(
            iterations=args.iterations,
            message=args.message,
            system_prompt=args.system_prompt,
            options=options,
        )
        candidate_report = candidate_runner.run_message_benchmark(
            iterations=args.iterations,
            message=args.message,
            system_prompt=args.system_prompt,
            options=candidate_options,
        )
        comparison = BenchmarkRunner.compare_reports(
            baseline_report,
            candidate_report,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
        )
        payload = comparison.to_dict()
        exit_code = 0 if (baseline_report.error_count == 0 and candidate_report.error_count == 0) else 1
    else:
        report = runner.run_message_benchmark(
            iterations=args.iterations,
            message=args.message,
            system_prompt=args.system_prompt,
            options=options,
        )
        payload = report.to_dict()
        exit_code = 0 if report.error_count == 0 else 1

    print(json.dumps(payload, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
