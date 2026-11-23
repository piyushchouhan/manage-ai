"""
Lesson 8.3: Streaming
======================

Run from backend/:   uv run python -m lessons.l83_streaming

Until now the browser asked, then waited for the WHOLE answer: 5-10s of "Thinking…" for a simple
read, then everything at once. The model writes word by word anyway; streaming hands those words on
as they come, so the first ones show up after ~1-2s.

Server-Sent Events (SSE) = one HTTP response that stays open; the server writes small events into it:

    event: token          <- "event" names the kind
    data: {"text": "There are"}
                          <- a blank line ends one event

Two streams:

  POST /api/chat/stream   one chat turn
      session   {session_id}
      tool      {names}       the agent is about to look something up ("get_stock")
      token     {text}        a piece of the answer, as the model writes it
      done      {reply}       the final reply (also saved in the chat)
      job       {job_id}      an investigation: follow it on the stream below

  GET /api/jobs/{id}/events   an investigation's progress, PUSHED instead of polled
      progress  {status, progress, reply}   on every change, until done/failed

How the tokens get out without touching the 7 layers of chat code: the tracer (1.4) is already passed
down to the agent loop. A StreamTracer has an on_token method; the loop (l12's generate) sees it and
calls the model with generate_content_stream instead. Everything else is unchanged.

How job progress gets out: the worker (8.2) is another process. Every time it updates the job row, it
also PUBLISHES it on Redis channel job:{id}. The API's SSE stream SUBSCRIBES to that channel.
Safety net: after 10 quiet seconds the stream re-reads the row, so a missed message only delays it.

Why the investigation's answer is NOT streamed token by token: it is checked (6.2) AFTER it is written.
Streaming it would show the user numbers that may then be rejected. Progress streams; the verified
answer arrives in one piece.

Experiments (a real server + worker, read over HTTP like the browser does):
  1. A simple read, streamed: time to the first token vs to the whole answer
  2. An investigation: the job's progress, pushed as it happens
"""

import json
import queue
import threading
import time
import traceback
import urllib.request
from collections.abc import Iterator

from lessons.common import header
from lessons.l14_tracing import Tracer
from lessons.l81_jobs import PUBLIC, PORT, bus, channel, ensure_tables, job_row, start_turn
from lessons.l82_queue import WORKER_CMD, start, start_api

DONE = ("done", "failed")
QUIET_S = 10  # no news for 10s: re-read the row (a missed message can't leave the stream stuck), keep alive


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# --- A chat turn, streamed ---------------------------------------------------------------------

class StreamTracer(Tracer):
    """1.4's tracer (still writes the trace file) + sends events to the browser."""

    def __init__(self, question: str, emit):
        super().__init__(question)
        self.emit = emit

    def on_token(self, text: str):  # l12's generate() streams when this exists
        self.emit("token", {"text": text})

    def model_call(self, step, response, seconds):
        super().model_call(step, response, seconds)
        if response.function_calls:  # the model decided to use tools: they run next
            self.emit("tool", {"names": [c.name for c in response.function_calls]})


def stream_turn(user_id: int, session_id: int, text: str, jobs) -> Iterator[str]:
    """The agent runs in a thread and puts events on a queue; this generator writes them out as they come.
    jobs: where investigations go (8.2's ArqQueue)."""
    events: queue.Queue = queue.Queue()

    def emit(kind: str, data: dict):
        events.put(sse(kind, data))

    def work():
        try:
            out = start_turn(user_id, session_id, text, jobs, tracer=StreamTracer(text, emit))
            if out.get("job_id"):
                emit("job", {"job_id": out["job_id"]})
            else:
                emit("done", {"reply": out["reply"]})
        except Exception:
            traceback.print_exc()
            emit("error", {"message": "Something went wrong on the server."})
        finally:
            events.put(None)  # end of stream

    threading.Thread(target=work, daemon=True).start()
    yield sse("session", {"session_id": session_id})
    while (event := events.get()) is not None:
        yield event


# --- A job's progress, pushed ----------------------------------------------------------------------

def current(job_id: int) -> dict:
    row = job_row(job_id)
    return {k: row[k] for k in PUBLIC}


def job_events(job_id: int) -> Iterator[str]:
    """Check ownership BEFORE calling this (the API does): a stream can't turn into a 404 halfway."""
    sub = bus.pubsub(ignore_subscribe_messages=True)
    sub.subscribe(channel(job_id))
    try:
        job = current(job_id)  # read AFTER subscribing: a change in between is not lost
        yield sse("progress", job)
        while job["status"] not in DONE:
            msg = sub.get_message(timeout=QUIET_S)
            if msg is not None:
                job = json.loads(msg["data"])
                yield sse("progress", job)
                continue
            # quiet: maybe a message was missed (a worker running older code, a Redis hiccup). The row is the truth.
            fresh = current(job_id)
            if fresh["status"] != job["status"] or fresh["progress"] != job["progress"]:
                job = fresh
                yield sse("progress", job)
            else:
                yield ": still working\n\n"  # an SSE comment: the browser ignores it; keeps proxies from cutting us off
    finally:
        sub.close()


# --- Experiments ------------------------------------------------------------------------------------

def read_stream(path: str, body: dict | None = None, user: int = 1) -> Iterator[tuple[str, dict]]:
    """Read an SSE response like the browser does: line by line, as it arrives."""
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", headers={"X-User-Id": str(user),
                                 "Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=300) as r:
        event = None
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                yield event, json.loads(line[6:])


def main():
    ensure_tables()
    api, worker = start_api(), start(WORKER_CMD, "worker")
    try:
        # --- Experiment 1 ---------------------------------------------------------
        header("Experiment 1: a simple read, streamed")
        t0, first, answer = time.perf_counter(), None, ""
        for event, data in read_stream("/api/chat/stream", {"message": "What is running low in Mumbai?"}):
            t = time.perf_counter() - t0
            if event == "token":
                first = first or t
                answer += data["text"]
                print(f"   [{t:5.2f}s] token  {data['text']!r}")
            else:
                print(f"   [{t:5.2f}s] {event:<6} {data}")
        print(f"\n   first token after {first:.1f}s, whole answer after {time.perf_counter() - t0:.1f}s"
              if first else "\n   no tokens streamed")

        # --- Experiment 2 ---------------------------------------------------------
        header("Experiment 2: an investigation, progress pushed (no polling)")
        t0, job_id = time.perf_counter(), None
        for event, data in read_stream("/api/chat/stream", {"message": "Why are we losing money on padlocks in BLR?"}):
            print(f"   [{time.perf_counter() - t0:5.1f}s] {event:<8} {data}")
            job_id = data.get("job_id", job_id)
        if job_id:
            for _, job in read_stream(f"/api/jobs/{job_id}/events"):
                print(f"   [{time.perf_counter() - t0:5.1f}s] progress {job['status']:<8} {job['progress'] or ''}")
            print(f"\n   reply:\n{job['reply'][:300]}...")
    finally:
        for p in (api, worker):
            p.terminate()
            p.wait()


if __name__ == "__main__":
    main()
