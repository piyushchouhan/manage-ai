"""
Lesson 8.2: A real job queue
=============================

Run from backend/:   uv run python -m lessons.l82_queue
The worker, for the real app:   uv run arq lessons.l82_queue.WorkerSettings

8.1's BackgroundTasks ran the investigation INSIDE the API process: restart the API (a deploy,
a crash, --reload saving a file) and every running job died with it.

Now there are three separate things:

    API process          POST /api/chat -> INSERT job row -> push "run_job(7)" onto a Redis list -> {job_id}
    Redis                the queue: the job waits here until a worker takes it
    worker process(es)   take a job from Redis -> run_job(7) (8.1's, unchanged) -> UPDATE the job row
                         and write progress on the row as it goes: "step 2/3: ..."

    UI                   polls GET /api/jobs/7 (the API reads the row)  -> status + progress + reply

So the API can restart while a job runs: the worker doesn't care, the row keeps updating, and the
new API process reads it. The worker can crash too: arq keeps the job in Redis until it FINISHES
(it locks it for timeout + 10s while running), so after a crash the job runs again from the start.
That means a job can run twice: run_job must be safe to repeat (it only saves the turn at the end).

The swap is small on purpose. 8.1's start_turn takes "anything with add_task(fn, *args)":
    8.1: BackgroundTasks()   add_task -> run in this process after the response
    8.2: ArqQueue()          add_task -> push onto Redis for a worker

Experiments (API and worker as real separate processes, logs in backend/traces/):
  1. No worker running: the job waits in Redis (queued). Start a worker: it picks it up.
  2. Restart the API in the middle of a job: the job keeps going, progress keeps moving.
"""

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import func
from redis.exceptions import RedisError

from app.config import REDIS_URL
from lessons.common import header
from lessons.l81_jobs import PORT, call, ensure_tables, fail_job, get_job, run_job

REDIS = RedisSettings.from_dsn(REDIS_URL)
JOB_TIMEOUT = 300  # seconds; investigations take ~2 min. Also how long a crashed job's lock lasts.
QUEUE_DOWN = "Sorry, the job queue is down, so I couldn't start this investigation. Please try again later."


# --- The API side: push the job onto Redis -------------------------------------------------------

async def enqueue(name: str, job_id: int):
    redis = await create_pool(REDIS)
    try:
        # _job_id: arq ignores a second push of the same id, so a job can't be queued twice
        await redis.enqueue_job(name, job_id, _job_id=f"job-{job_id}")
    finally:
        await redis.aclose()


class ArqQueue:
    """Same add_task() as FastAPI's BackgroundTasks, but the work goes to Redis for a worker process."""

    def add_task(self, fn, job_id: int):
        try:
            asyncio.run(enqueue(fn.__name__, job_id))  # the endpoint is a plain def: no event loop running here
        except (OSError, RedisError) as e:
            print(f"   [queue] can't reach Redis: {e}")
            fail_job(job_id, QUEUE_DOWN)  # the UI's next poll shows this, instead of 'queued' forever


# --- The worker side ---------------------------------------------------------------------------------

async def run_job_async(ctx, job_id: int):
    """arq runs coroutines; run_job is plain blocking code (DB, model calls), so it runs in a thread."""
    await asyncio.to_thread(run_job, job_id)


class WorkerSettings:
    functions = [func(run_job_async, name="run_job", timeout=JOB_TIMEOUT)]  # name = what the API pushes
    redis_settings = REDIS
    max_jobs = 4  # investigations at the same time, per worker process


# --- Experiments: real processes ---------------------------------------------------------------------

LOGS = Path("traces")
API_CMD = [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)]
WORKER_CMD = [shutil.which("arq") or "arq", "lessons.l82_queue.WorkerSettings"]


def start(cmd: list[str], name: str) -> subprocess.Popen:
    LOGS.mkdir(exist_ok=True)
    log = open(LOGS / f"l82_{name}.log", "a")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def start_api() -> subprocess.Popen:
    proc = start(API_CMD, "api")
    for _ in range(100):
        try:
            if call("/api/health")[0] == 200:
                return proc
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("API didn't start: see traces/l82_api.log")


def watch(job_id: int, t0: float, until=lambda j: j["status"] in ("done", "failed"), via_api: bool = True) -> dict:
    """Poll like the UI (via_api) or read the row directly (when the API is down). Print each change."""
    last = None
    while True:
        j = call(f"/api/jobs/{job_id}")[1] if via_api else get_job(1, job_id)
        now = (j["status"], j["progress"])
        if now != last:
            print(f"   [{time.perf_counter() - t0:5.1f}s] {'API' if via_api else 'DB '}  {j['status']:<8} {j['progress'] or ''}")
            last = now
        if until(j):
            return j
        time.sleep(1)


def main():
    ensure_tables()
    print(f"   logs: {LOGS.resolve()}/l82_api.log, l82_worker.log")
    api, worker = start_api(), None
    try:
        # --- Experiment 1 ---------------------------------------------------------
        header("Experiment 1: no worker running -> the job waits in Redis")
        t0 = time.perf_counter()
        _, r = call("/api/chat", {"message": "Why are we losing money on padlocks in BLR?"})
        print(f"   response after {time.perf_counter() - t0:.1f}s: path={r['path']} job_id={r['job_id']}")
        job_id = r["job_id"]
        watch(job_id, t0, until=lambda j: time.perf_counter() - t0 > 8)
        print("   ...still queued after 8s: nobody is taking jobs. Starting a worker.")
        worker = start(WORKER_CMD, "worker")
        j = watch(job_id, t0)
        print(f"\n   reply ({j['seconds']}s after the question):\n{j['reply'][:400]}...")

        # --- Experiment 2 ---------------------------------------------------------
        header("Experiment 2: restart the API in the middle of a job")
        t0 = time.perf_counter()
        _, r = call("/api/chat", {"message": "Why do some of our products keep running low?"})
        job_id = r["job_id"]
        watch(job_id, t0, until=lambda j: (j["progress"] or "").startswith("step 1"))
        api.terminate()
        api.wait()
        print("   >>> API STOPPED. The UI's polls would fail now; reading the row straight from the DB:")
        watch(job_id, t0, until=lambda j: (j["progress"] or "").startswith("step 2") or j["status"] == "done",
              via_api=False)
        api = start_api()
        print("   >>> API STARTED again (a new process). Polling through it:")
        j = watch(job_id, t0)
        print(f"\n   final status: {j['status']} after {j['seconds']}s")
        print(f"   reply:\n{j['reply'][:400]}...")
    finally:
        for p in (api, worker):
            if p:
                p.terminate()
                p.wait()


if __name__ == "__main__":
    main()
