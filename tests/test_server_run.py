#!/usr/bin/env python3
"""Live API server smoke test.

This module serves two purposes:
- It is a pytest test collected during normal test runs.
- It can be run directly as a script for manual verification.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import uvicorn


# Allow execution from repository root without package installation.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from ollama_wrapper.api_server import ChatSessionManager, create_app


class DummyWrapper:
    async def ask_async(self, message: str, metadata_filter: dict[str, Any] | None = None) -> str:
        return f"echo: {message}"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _wait_until_ready(base_url: str, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            _request_json(f"{base_url}/openapi.json")
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            time.sleep(0.2)
    raise TimeoutError(f"Server did not become ready within {timeout_sec:.1f}s")


def run_server_smoke_test(host: str, port: int, startup_timeout: float = 8.0) -> None:
    wrapper = DummyWrapper()
    manager = ChatSessionManager(wrapper)
    app = create_app(manager)

    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config=config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"

    try:
        _wait_until_ready(base_url, timeout_sec=startup_timeout)

        created = _request_json(
            f"{base_url}/session",
            method="POST",
            payload={"system_prompt": "You are helpful.", "metadata": {"test": "true"}},
        )
        session_id = created["session_id"]

        replied = _request_json(
            f"{base_url}/session/{session_id}/message",
            method="POST",
            payload={"message": "hello"},
        )
        assert replied["reply"] == "echo: hello", f"Unexpected reply: {replied!r}"

        state = _request_json(f"{base_url}/session/{session_id}")
        assert state["session_id"] == session_id
    finally:
        server.should_exit = True
        thread.join(timeout=3)


def test_server_run_smoke() -> None:
    run_server_smoke_test(host="127.0.0.1", port=_find_free_port(), startup_timeout=8.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live API server smoke test.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Server port to bind.")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for server startup.",
    )
    args = parser.parse_args()

    try:
        run_server_smoke_test(host=args.host, port=args.port, startup_timeout=args.startup_timeout)
        print("PASS: server started and endpoints responded correctly")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())