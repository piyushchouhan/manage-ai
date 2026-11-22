"""Chat API for the React frontend.

Run from backend/:   uv run uvicorn app.main:app --reload --port 8000
Then the frontend:   cd frontend && npm run dev   ->  http://localhost:5173

Contract (what frontend/src/App.tsx sends and expects). Every call carries the header
X-User-Id: <id>  (a stand-in for a real login; see below).
    GET  /api/users                   -> [{"id", "name", "role", "warehouses"}]
    POST /api/chat                    {"session_id": int | null, "message": "..."}
                                   -> {"session_id": int, "reply": "..."}
    GET  /api/sessions/{id}/messages -> [{"role": "user"|"assistant", "content": "..."}, ...]

The agent is the latest lesson's: 1.2's loop + 3.3's tools, then module 4:
history in the DB (4.1), selections (4.2), compaction (4.3), token budgets (4.4),
tool/history selection (4.5), long-term memory + logbook search (4.6),
and isolation per user and session (4.7).

NOT secure yet: the browser just SAYS who it is (X-User-Id), so anyone can claim any user.
A real login (signed session cookie / token) replaces the header later; everything below
already uses the user id only through code, never through the model.
"""

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.db import query
from lessons.l14_tracing import Tracer
from lessons.l41_history import new_session
from lessons.l47_isolation import NotYourSession, chat as chat_turn, ensure_tables, messages_for, owner_of

ensure_tables()
app = FastAPI(title="Manage AI")


class ChatRequest(BaseModel):
    session_id: int | None = None  # None = start a new conversation
    message: str


class ChatReply(BaseModel):
    session_id: int
    reply: str


@app.get("/api/users")
def users() -> list[dict]:
    return query(
        """SELECT u.id, u.name, u.role, array_agg(w.code ORDER BY w.code) AS warehouses
           FROM users u JOIN user_warehouse_access a ON a.user_id = u.id
           JOIN warehouses w ON w.id = a.warehouse_id GROUP BY u.id ORDER BY u.id"""
    )


@app.post("/api/chat", response_model=ChatReply)
def chat(req: ChatRequest, x_user_id: int = Header(1)) -> ChatReply:
    # Plain `def` (not async): FastAPI runs it in a thread, so a slow model call
    # doesn't block other requests.
    sid = req.session_id
    if sid is None or owner_of(sid) != x_user_id:  # unknown, or someone else's: start a new chat
        sid = new_session(x_user_id)
    reply = chat_turn(x_user_id, sid, req.message, tracer=Tracer(req.message))  # trace saved to backend/traces/
    return ChatReply(session_id=sid, reply=reply)


@app.get("/api/sessions/{session_id}/messages")
def messages(session_id: int, x_user_id: int = Header(1)) -> list[dict]:
    try:
        return messages_for(x_user_id, session_id)
    except NotYourSession:
        raise HTTPException(404, "no such session")  # 404, not 403: don't reveal it exists


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
