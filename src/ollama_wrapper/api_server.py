import asyncio
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    system_prompt: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = None


class MessageRequest(BaseModel):
    message: str


class ChatSession:
    def __init__(self, session_id: str, system_prompt: str = "", metadata: dict = None):
        self.session_id = session_id
        self.system_prompt = system_prompt or ""
        self.metadata = metadata or {}
        self.chat_history = []
        self.running_summary = ""
        self.created_at = asyncio.get_event_loop().time()


class ChatSessionManager:
    """Manage chat sessions and per-session context for multiple agents/connections."""

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.sessions: Dict[str, ChatSession] = {}
        self.lock = asyncio.Lock()

    async def create_session(self, system_prompt: str = "", metadata: dict = None) -> ChatSession:
        async with self.lock:
            sid = str(uuid.uuid4())
            sess = ChatSession(sid, system_prompt=system_prompt, metadata=metadata)
            self.sessions[sid] = sess
            return sess

    async def get_session(self, session_id: str) -> ChatSession:
        sess = self.sessions.get(session_id)
        if not sess:
            raise KeyError(f"Session {session_id} not found")
        return sess

    async def close_session(self, session_id: str) -> None:
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]


def create_app(manager: ChatSessionManager) -> FastAPI:
    app = FastAPI()

    @app.post("/session")
    async def create(req: SessionCreateRequest):
        sess = await manager.create_session(system_prompt=req.system_prompt, metadata=req.metadata)
        return {"session_id": sess.session_id}

    @app.post("/session/{session_id}/message")
    async def message(session_id: str, req: MessageRequest):
        try:
            sess = await manager.get_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found")

        # Append user message
        sess.chat_history.append({"role": "user", "content": req.message})

        # Use the provided OllamaWrapper instance to ask asynchronously
        try:
            if hasattr(manager.wrapper, "ask_async"):
                resp = await manager.wrapper.ask_async(req.message, metadata_filter=sess.metadata)
            else:
                # fallback to sync ask
                loop = asyncio.get_running_loop()
                resp = await loop.run_in_executor(None, manager.wrapper.ask, req.message, sess.metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Append assistant reply
        sess.chat_history.append({"role": "assistant", "content": resp})

        return {"reply": resp}

    @app.get("/session/{session_id}")
    async def session_state(session_id: str):
        try:
            sess = await manager.get_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found")
        return {
            "session_id": sess.session_id,
            "system_prompt": sess.system_prompt,
            "metadata": sess.metadata,
            "chat_history": sess.chat_history,
            "running_summary": sess.running_summary,
        }

    @app.delete("/session/{session_id}")
    async def close(session_id: str):
        await manager.close_session(session_id)
        return {"closed": True}

    return app
