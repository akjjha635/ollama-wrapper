import json
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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


def _wait_until_ready(proc: subprocess.Popen[str], base_url: str, timeout_sec: float = 12.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"Launcher exited early with code {proc.returncode}")
        try:
            _request_json(f"{base_url}/openapi.json")
            return
        except urllib.error.URLError:
            time.sleep(0.2)

    raise TimeoutError(f"Server did not become ready within {timeout_sec:.1f}s")


def test_start_chat_server_launcher_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "start_chat_server.py"
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [
            sys.executable,
            str(script_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dummy-wrapper",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout = ""
    stderr = ""
    try:
        _wait_until_ready(proc, base_url)

        created = _request_json(
            f"{base_url}/session",
            method="POST",
            payload={"system_prompt": "You are helpful."},
        )
        session_id = created["session_id"]

        replied = _request_json(
            f"{base_url}/session/{session_id}/message",
            method="POST",
            payload={"message": "ping"},
        )
        assert replied["reply"] == "echo: ping"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            stdout, stderr = proc.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=5)

    assert proc.returncode == 0, (
        "Launcher script did not exit cleanly.\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )
