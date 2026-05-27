#!/usr/bin/env python3
"""Start the Ollama wrapper chat API server for local development."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from typing import Any

import uvicorn

# Allow running directly from repository root without package install.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from ollama_wrapper import OllamaWrapper
from ollama_wrapper.api_server import ChatSessionManager, create_app


class DummyDevWrapper:
    """Minimal wrapper used for local integration tests without Ollama."""

    def __init__(self) -> None:
        self._api_server_task: asyncio.Task[Any] | None = None

    async def ask_async(self, message: str, metadata_filter: dict[str, Any] | None = None) -> str:
        return f"echo: {message}"

    async def start_api_server(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        if self._api_server_task is not None and not self._api_server_task.done():
            return

        manager = ChatSessionManager(self)
        app = create_app(manager)
        config = uvicorn.Config(app=app, host=host, port=port, loop="asyncio", log_level="info")
        server = uvicorn.Server(config=config)

        async def _runner() -> None:
            await server.serve()

        self._api_server_task = asyncio.get_running_loop().create_task(_runner())

    async def stop_api_server(self) -> None:
        if self._api_server_task is None:
            return
        self._api_server_task.cancel()
        try:
            await self._api_server_task
        except asyncio.CancelledError:
            pass
        self._api_server_task = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the chat session API server for development and integration testing."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the API server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the API server.")

    parser.add_argument(
        "--ollama-host",
        default="localhost",
        help="Ollama daemon host used by OllamaWrapper.",
    )
    parser.add_argument(
        "--ollama-port",
        type=int,
        default=11434,
        help="Ollama daemon port used by OllamaWrapper.",
    )

    parser.add_argument("--connection-type", choices=["sync", "async"], default="async")
    parser.add_argument("--llm-model", default="deepseek-r1:1.5b")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--db-storage-path", default="./local_vector_db")
    parser.add_argument("--max-active-turns", type=int, default=4)
    parser.add_argument(
        "--retrieval-backend",
        choices=["linear", "faiss"],
        default="linear",
        help="Dense retrieval backend for core retriever path.",
    )
    parser.add_argument(
        "--dummy-wrapper",
        action="store_true",
        help="Run with a local dummy echo backend (no Ollama required).",
    )

    return parser.parse_args()


async def run_server(args: argparse.Namespace) -> None:
    if args.dummy_wrapper:
        wrapper: Any = DummyDevWrapper()
    else:
        wrapper = OllamaWrapper(
            ip=args.ollama_host,
            port=args.ollama_port,
            connection_type=args.connection_type,
            llm_model=args.llm_model,
            embed_model=args.embed_model,
            db_storage_path=args.db_storage_path,
            max_active_turns=args.max_active_turns,
            retrieval_backend=args.retrieval_backend,
        )

    await wrapper.start_api_server(host=args.host, port=args.port)
    print(f"Chat API server running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")

    stop_event = asyncio.Event()

    def _request_shutdown() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _request_shutdown)
        loop.add_signal_handler(signal.SIGTERM, _request_shutdown)
    except NotImplementedError:
        # Signal handlers are not available on some platforms/event loops.
        pass

    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        await wrapper.stop_api_server()


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run_server(args))
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to start chat server: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
