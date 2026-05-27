#!/usr/bin/env python3
"""Convert benchmark comparison JSON into a markdown summary table."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from ollama_wrapper.eval import comparison_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render benchmark comparison JSON as markdown.")
    parser.add_argument("--input", required=True, help="Path to comparison JSON produced by run_benchmark.py --compare")
    parser.add_argument("--output", default="", help="Optional output markdown file path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in input file: {exc}", file=sys.stderr)
        return 2

    markdown = comparison_to_markdown(payload)
    print(markdown)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
