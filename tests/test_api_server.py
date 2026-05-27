import asyncio

from fastapi.testclient import TestClient
from ollama_wrapper.api_server import ChatSessionManager, create_app


class DummyWrapper:
    async def ask_async(self, message, metadata_filter=None):
        return f"echo: {message}"


def test_session_create_and_message():
    wrapper = DummyWrapper()
    manager = ChatSessionManager(wrapper)
    app = create_app(manager)

    client = TestClient(app)

    # create session
    resp = client.post("/session", json={"system_prompt": "You are helpful."})
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    # send message
    r2 = client.post(f"/session/{sid}/message", json={"message": "hello"})
    assert r2.status_code == 200
    assert r2.json()["reply"] == "echo: hello"

    # get state
    r3 = client.get(f"/session/{sid}")
    assert r3.status_code == 200
    state = r3.json()
    assert state["session_id"] == sid
    assert any(m["content"] == "hello" for m in state["chat_history"]) 
