"""Chat API for the React frontend.

Run from backend/:   uv run uvicorn app.main:app --reload --port 8000
Then the frontend:   cd frontend && npm run dev   ->  http://localhost:5173

Contract (what frontend/src/App.tsx sends and expects):
    POST /api/chat                    {"session_id": int | null, "message": "..."}
                                   -> {"session_id": int, "reply": "..."}
    GET  /api/sessions/{id}/messages -> [{"role": "user"|"assistant", "content": "..."}, ...]

History lives in the DB (lesson 4.1), not in the browser: the browser only keeps the session id.
The last list shown and any pending question are saved too (4.2), so "the 2nd one" works.
Long chats are compacted (4.3): old turns become a running summary, recent ones are sent word for word.
The agent is the latest one from the lessons: 1.2's loop + 3.3's tools (asks when a product name is ambiguous).
When a later lesson improves the agent, change the imports below.

Not yet (coming in later lessons): login/permissions (2.1 is ready, needs a
user from the session), streaming (8.3).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lessons.l14_tracing import Tracer
from lessons.l41_history import load_messages, new_session, session_exists
from lessons.l43_compaction import chat as chat_turn  # 4.1 history + 4.2 selections + 4.3 compaction
from lessons.l43_compaction import ensure_tables

ensure_tables()
app = FastAPI(title="Manage AI")


class ChatRequest(BaseModel):
    session_id: int | None = None  # None = start a new conversation
    message: str


class ChatReply(BaseModel):
    session_id: int
    reply: str


@app.post("/api/chat", response_model=ChatReply)
def chat(req: ChatRequest) -> ChatReply:
    # Plain `def` (not async): FastAPI runs it in a thread, so a slow model call
    # doesn't block other requests.
    sid = req.session_id if req.session_id and session_exists(req.session_id) else new_session()
    reply = chat_turn(sid, req.message, tracer=Tracer(req.message))  # trace saved to backend/traces/
    return ChatReply(session_id=sid, reply=reply)


@app.get("/api/sessions/{session_id}/messages")
def messages(session_id: int) -> list[dict]:
    if not session_exists(session_id):
        raise HTTPException(404, "no such session")
    return load_messages(session_id)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
