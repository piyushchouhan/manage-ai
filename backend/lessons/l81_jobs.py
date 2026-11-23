"""
Lesson 8.1: Background jobs (simple)
=====================================

Run from backend/:   uv run python -m lessons.l81_jobs

7.3's investigation takes 1-2 minutes. Holding an HTTP request open that long is fragile:
proxies time out after ~60s, a refresh loses the answer, and the user stares at "Thinking…".

So the chat answers in two speeds:

    POST /api/chat
      router (~1-2s, cheap model)            5.2's choose(): decide BEFORE doing the work
        smalltalk / clarify / simple_read    run now, reply in the response       (seconds)
        investigation                        INSERT a row in jobs (status queued)
                                             return {job_id} immediately          (~2s)
                                             BackgroundTasks runs run_job AFTER the response is sent
    GET /api/jobs/{id}                       the UI polls: queued -> running -> done (+ reply)

    run_job:  running -> 7.3's investigate -> save the turn in the chat -> done (or failed)

The jobs table is the hand-off: the request that started the job and the requests that poll it
never share memory, only this row.

The weakness, on purpose: BackgroundTasks runs INSIDE the API process. Restart the server and every
running job dies with it. fail_stale() at startup at least marks them failed instead of "running"
forever. 8.2 moves jobs to a separate worker process with a real queue.

Since 8.2, app.main sends jobs to Redis instead of BackgroundTasks: experiment 2 below then needs a
worker running (uv run arq lessons.l82_queue.WorkerSettings).

Experiments (a real uvicorn server on :8001, called over HTTP like the UI does):
  1. A simple read: the answer comes in the response
  2. An investigation: {job_id} comes back at once, then polling until done
  3. Another user polls that job: 404
  4. A "server restart" while a job runs: fail_stale()
"""

import json
import threading
import time
import traceback
import urllib.error
import urllib.request

import redis
import uvicorn
from sqlalchemy import func, text, update

from app.config import REDIS_URL
from app.db import SessionLocal, engine
from app.models import Base, Job
from lessons.common import header
from lessons.l41_history import new_session, save_turn
from lessons.l47_isolation import ensure_tables as ensure_chat_tables
from lessons.l52_dispatch import choose
from lessons.l73_investigation import HANDLERS_V2

FAILED = "Sorry, the investigation failed before it finished. Please ask again."
RESTARTED = "Sorry, the server restarted while this investigation was running. Please ask again."


def ensure_tables():
    ensure_chat_tables()
    Base.metadata.create_all(engine, tables=[Job.__table__])
    with engine.begin() as conn:  # jobs existed before 8.2: add the progress column
        conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS progress text"))


# --- The jobs table ---------------------------------------------------------------------------

def create_job(user_id: int, session_id: int, question: str) -> int:
    with SessionLocal() as db:
        job = Job(user_id=user_id, session_id=session_id, question=question, status="queued")
        db.add(job)
        db.commit()
        return job.id


def set_job(job_id: int, status: str, reply: str | None = None):
    values = {"status": status, "reply": reply}
    if status in ("done", "failed"):
        values["finished_at"] = func.now()
    with SessionLocal() as db:
        db.execute(update(Job).where(Job.id == job_id).values(**values))
        db.commit()
    notify(job_id)


def set_progress(job_id: int, message: str):
    with SessionLocal() as db:
        db.execute(update(Job).where(Job.id == job_id).values(progress=message))
        db.commit()
    notify(job_id)


PUBLIC = ("id", "question", "status", "reply", "progress", "seconds")  # what the owner may see of a job
bus = redis.Redis.from_url(REDIS_URL)  # connects on first use, not at import


def channel(job_id: int) -> str:
    return f"job:{job_id}"


def notify(job_id: int):
    """8.3: tell anyone listening (an open SSE stream) that the row changed. Best effort: if Redis is
    down, polling /api/jobs/{id} still works, because the row is the truth."""
    try:
        row = job_row(job_id)
        bus.publish(channel(job_id), json.dumps({k: row[k] for k in PUBLIC}, default=str))
    except redis.RedisError:
        pass


def fail_job(job_id: int, message: str):
    """The reply goes into the chat too, so the conversation shows what happened."""
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        session_id, question = job.session_id, job.question
    save_turn(session_id, question, message)
    set_job(job_id, "failed", message)


def job_row(job_id: int):
    with SessionLocal() as db:
        row = db.execute(text(
            """SELECT id, user_id, question, status, reply, progress,
                      EXTRACT(EPOCH FROM COALESCE(finished_at, now()) - created_at)::int AS seconds
               FROM jobs WHERE id = :id"""), {"id": job_id}).mappings().first()
    return row


def get_job(user_id: int, job_id: int) -> dict | None:
    """None if it doesn't exist OR isn't this user's (the API turns both into the same 404)."""
    row = job_row(job_id)
    if row is None or row["user_id"] != user_id:
        return None
    return {k: row[k] for k in PUBLIC}


# --- Running a job (after the response has gone) ------------------------------------------------

def run_job(job_id: int):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        user_id, session_id, question = job.user_id, job.session_id, job.question
    set_job(job_id, "running")
    try:
        reply = HANDLERS_V2["investigation"](user_id, session_id, question, None,  # 7.3: also saves the turn
                                             progress=lambda p: set_progress(job_id, p))
        set_job(job_id, "done", reply)
    except Exception:
        traceback.print_exc()  # the details stay in the server log, not in the chat
        fail_job(job_id, FAILED)


def fail_stale() -> int:
    """At startup: jobs still queued/running belonged to a process that is gone. -> how many."""
    with SessionLocal() as db:
        stale = [j.id for j in db.query(Job).filter(Job.status.in_(["queued", "running"]))]
    for job_id in stale:
        fail_job(job_id, RESTARTED)
    return len(stale)


# --- One chat turn, two speeds ----------------------------------------------------------------------

def start_turn(user_id: int, session_id: int, text: str, background, tracer=None,
               handlers: dict = HANDLERS_V2) -> dict:
    """background: anything with add_task(fn, *args), e.g. FastAPI's BackgroundTasks.
    handlers: the quick paths (9.2 adds tables and follow-ups to simple reads).
    -> {"path", "reply"} for a quick path, or {"path", "job_id"} for an investigation."""
    path, d = choose(user_id, session_id, text, tracer)
    if path == "investigation":
        job_id = create_job(user_id, session_id, text)
        background.add_task(run_job, job_id)  # runs AFTER this response is sent
        return {"path": path, "job_id": job_id}
    return {"path": path, "reply": handlers[path](user_id, session_id, text, d, tracer=tracer)}


# --- Experiments ------------------------------------------------------------------------------------

ASHA, RAVI = 1, 4
PORT = 8001


def serve() -> uvicorn.Server:
    """The real API in a thread, so the experiments go over HTTP like the UI does."""
    server = uvicorn.Server(uvicorn.Config("app.main:app", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return server


def call(path: str, body: dict | None = None, user: int = ASHA) -> tuple[int, dict | list]:
    """One HTTP request, like the UI's fetch(): POST with a body, else GET. -> (status, JSON)."""
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", headers={"X-User-Id": str(user),
                                 "Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    ensure_tables()
    server = serve()
    try:
        # --- Experiment 1 ---------------------------------------------------------
        header("Experiment 1: a simple read -> the answer is in the response")
        t0 = time.perf_counter()
        _, r = call("/api/chat", {"message": "How many padlocks are in Bangalore?"})
        print(f"   response after {time.perf_counter() - t0:.1f}s: path={r['path']} job_id={r['job_id']}")
        print(f"   reply: {r['reply']}")
        sid = r["session_id"]

        # --- Experiment 2 ---------------------------------------------------------
        header("Experiment 2: an investigation -> {job_id} at once, then poll")
        t0 = time.perf_counter()
        _, r = call("/api/chat", {"session_id": sid, "message": "Why are we losing money on padlocks in BLR?"})
        print(f"   response after {time.perf_counter() - t0:.1f}s: path={r['path']} job_id={r['job_id']} reply={r['reply']}")
        job_id, last = r["job_id"], None
        while True:
            _, j = call(f"/api/jobs/{job_id}")
            if j["status"] != last:
                print(f"   [{time.perf_counter() - t0:5.1f}s] poll -> status={j['status']}")
                last = j["status"]
            if j["status"] in ("done", "failed"):
                break
            time.sleep(2)  # the UI polls every 2s too
        print(f"\n   reply after {j['seconds']}s:\n{j['reply']}")
        _, saved = call(f"/api/sessions/{sid}/messages")
        print(f"\n   the chat history now has {len(saved)} messages; the last one is the investigation's reply: "
              f"{saved[-1]['content'] == j['reply']}")

        # --- Experiment 3 ---------------------------------------------------------
        header("Experiment 3: Ravi polls Asha's job")
        code, body = call(f"/api/jobs/{job_id}", user=RAVI)
        print(f"   GET /api/jobs/{job_id} as Ravi -> {code} {body}")

        # --- Experiment 4 ---------------------------------------------------------
        header("Experiment 4: the server 'restarts' while a job is running")
        stuck = create_job(ASHA, new_session(ASHA), "Why do some of our products keep running low?")
        set_job(stuck, "running")  # as if run_job had started, then the process died
        print(f"   before: {get_job(ASHA, stuck)}")
        print(f"   fail_stale() at startup -> {fail_stale()} job(s) marked failed")
        print(f"   after:  {get_job(ASHA, stuck)}")
    finally:
        server.should_exit = True


if __name__ == "__main__":
    main()
