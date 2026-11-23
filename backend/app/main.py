"""Chat API for the React frontend.

Run from backend/:   uv run uvicorn app.main:app --reload --port 8000
Then the frontend:   cd frontend && npm run dev   ->  http://localhost:5173

Contract (what frontend/src/App.tsx sends and expects). Every call carries the header
X-User-Id: <id>  (a stand-in for a real login; see below).
    GET  /api/users                   -> [{"id", "name", "role", "warehouses"}]
    POST /api/chat                    {"session_id": int | null, "message": "..."}
                                   -> {"session_id", "path", "reply": "..."}           quick paths
                                   -> {"session_id", "path", "job_id": int}             investigations (8.1)
    GET  /api/jobs/{id}               -> {"id", "question", "status": queued|running|done|failed,
                                          "progress": "step 2/3: ...", "reply", "seconds"}
    GET  /api/sessions/{id}/messages -> [{"role": "user"|"assistant", "content": "...",
                                           "display": {"blocks", "follow_ups"} | null}, ...]   (display: 9.2)
  Streaming (8.3, Server-Sent Events; what the UI uses):
    POST /api/chat/stream             same body -> events: session, tool, token..., display, then done or job
    GET  /api/jobs/{id}/events        -> events: progress (on every change, until done/failed)

The agent is the latest lesson's: 1.2's loop + 3.3's tools, then module 4:
history in the DB (4.1), selections (4.2), compaction (4.3), token budgets (4.4),
tool/history selection (4.5), long-term memory + logbook search (4.6),
and isolation per user and session (4.7). Since 8.1, 5.2's router decides first: quick paths answer
in the response, investigations (7.3: planner -> workers -> validator) run as a background job.
Since 8.2 that job runs in a separate WORKER process via Redis, so it survives an API restart:
    uv run arq lessons.l82_queue.WorkerSettings

NOT secure yet: the browser just SAYS who it is (X-User-Id), so anyone can claim any user.
A real login (signed session cookie / token) replaces the header later; everything below
already uses the user id only through code, never through the model.
"""

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.db import query
from lessons.l14_tracing import Tracer
from lessons.l41_history import new_session
from lessons.l47_isolation import NotYourSession, messages_for, owner_of
from lessons.l81_jobs import get_job, start_turn
from lessons.l82_queue import ArqQueue
from lessons.l83_streaming import job_events, stream_turn
from lessons.l92_rendering import HANDLERS_V3, ensure_tables

ensure_tables()  # no fail_stale() any more (8.1): running jobs belong to the worker, not to this process
queue = ArqQueue()
app = FastAPI(title="Manage AI")


class ChatRequest(BaseModel):
    session_id: int | None = None  # None = start a new conversation
    message: str


class ChatReply(BaseModel):
    session_id: int
    path: str                   # which 5.2 handler ran: smalltalk, clarify, simple_read, pending, investigation
    reply: str | None = None    # quick paths
    job_id: int | None = None   # investigations: poll /api/jobs/{job_id}


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
    sid = session_for(x_user_id, req.session_id)
    out = start_turn(x_user_id, sid, req.message, queue, tracer=Tracer(req.message), handlers=HANDLERS_V3)  # traces: backend/traces/
    return ChatReply(session_id=sid, **out)


def session_for(user_id: int, session_id: int | None) -> int:
    if session_id is None or owner_of(session_id) != user_id:  # unknown, or someone else's: start a new chat
        return new_session(user_id)
    return session_id


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}  # don't let proxies hold events back


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest, x_user_id: int = Header(1)) -> StreamingResponse:
    sid = session_for(x_user_id, req.session_id)
    return StreamingResponse(stream_turn(x_user_id, sid, req.message, queue, HANDLERS_V3),
                             media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/jobs/{job_id}/events")
def job_stream(job_id: int, x_user_id: int = Header(1)) -> StreamingResponse:
    if get_job(x_user_id, job_id) is None:  # check BEFORE the stream starts: after, there's no status code
        raise HTTPException(404, "no such job")
    return StreamingResponse(job_events(job_id), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/jobs/{job_id}")
def job(job_id: int, x_user_id: int = Header(1)) -> dict:
    j = get_job(x_user_id, job_id)
    if j is None:
        raise HTTPException(404, "no such job")  # someone else's looks the same as missing
    return j


@app.get("/api/sessions/{session_id}/messages")
def messages(session_id: int, x_user_id: int = Header(1)) -> list[dict]:
    try:
        return messages_for(x_user_id, session_id)
    except NotYourSession:
        raise HTTPException(404, "no such session")  # 404, not 403: don't reveal it exists


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
