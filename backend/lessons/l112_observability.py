"""
Lesson 11.2: Observability and cost
====================================

Run from backend/:   uv run python -m lessons.l112_observability             (report from what's recorded)
                     uv run python -m lessons.l112_observability --traffic   (first run 11.1's 25 questions)

1.4's trace files answer "what happened in THIS request?". Running a product needs "what happens ON
AVERAGE?": what does a simple read cost, how slow is an investigation, which model eats the budget.

Two tables (Postgres, like everything else):
    agent_runs   one row per request: route, user, question, seconds, ok
    llm_calls    one row per model call: run_id, model, tokens (prompt / cached / output / thinking),
                 latency, cost in USD

How the calls get recorded without touching the ~20 places that call the model: every one of them does
client.models.generate_content(...) on the SAME client object. install() swaps those two methods for
versions that call the original and then write an llm_calls row. Which request a call belongs to comes
from a context variable set by observe(): start_turn and run_job open one, and every model call made
inside (router, agent steps, planner, workers, synthesizer, layout) is counted against it.

    cost = (prompt - cached) x input price + cached x cache price + (output + thinking) x output price

Prices are per 1M tokens, paid tier, standard (aistudio.google.com/docs/pricing, read 23/11/2026).
gemini-3.8-flash has an introductory price until 31/12/2026: from 01/01/2027 the same traffic costs 2x.

Langfuse (or Phoenix) would give the same numbers in a web UI, plus trace trees. It needs an account
(data leaves your machine) or a self-hosted stack; the measuring idea is exactly this.

NOT counted: embeddings (search_notes, memory search) go through embed_content, which isn't wrapped.
"""

import argparse
import contextvars
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date

from sqlalchemy import text

from app.db import SessionLocal, engine
from app.llm import client
from app.models import AgentRun, Base, LlmCall
from lessons.common import header


# --- Prices -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Price:
    input: float   # USD per 1M prompt tokens
    output: float  # USD per 1M output tokens, thinking included
    cached: float  # USD per 1M cached prompt tokens


PRICES: dict[str, list[tuple[date | None, Price]]] = {  # (valid until, price); None = from then on
    "gemini-3.8-flash": [(date(2026, 12, 31), Price(0.75, 3.75, 0.075)), (None, Price(1.50, 7.50, 0.15))],
    "gemini-3.5-flash-lite": [(None, Price(0.30, 2.50, 0.03))],
}


def price_for(model: str, on: date | None = None) -> Price | None:
    on = on or date.today()
    for until, price in PRICES.get(model, []):
        if until is None or on <= until:
            return price
    return None


def cost(model: str, prompt: int, cached: int, output: int, thinking: int, on: date | None = None) -> float | None:
    p = price_for(model, on)
    if p is None:
        return None
    return ((prompt - cached) * p.input + cached * p.cached + (output + thinking) * p.output) / 1_000_000


# --- Which request is running (set by observe, read by every model call) -----------------------------

RUN: contextvars.ContextVar[dict | None] = contextvars.ContextVar("run", default=None)


def ensure_tables():
    Base.metadata.create_all(engine, tables=[AgentRun.__table__, LlmCall.__table__])


@contextmanager
def observe(user_id: int | None, question: str, route: str = "?"):
    """Everything inside is one request. Set run["route"] once you know it."""
    run = {"id": uuid.uuid4().hex[:16], "route": route}
    token, t0, ok = RUN.set(run), time.perf_counter(), False
    try:
        yield run
        ok = True
    finally:
        RUN.reset(token)
        try:
            with SessionLocal() as db:
                db.add(AgentRun(id=run["id"], route=run["route"], user_id=user_id, question=question,
                                seconds=round(time.perf_counter() - t0, 2), ok=ok))
                db.commit()
        except Exception as e:  # measuring must never break the request
            print(f"   [observe] couldn't save the run: {e!r}")


def record(model: str, usage, seconds: float):
    try:
        u = {"prompt": usage.prompt_token_count or 0, "cached": usage.cached_content_token_count or 0,
             "output": usage.candidates_token_count or 0, "thinking": usage.thoughts_token_count or 0} \
            if usage else {"prompt": 0, "cached": 0, "output": 0, "thinking": 0}
        run = RUN.get()
        with SessionLocal() as db:
            db.add(LlmCall(run_id=run["id"] if run else None, model=model, prompt_tokens=u["prompt"],
                           cached_tokens=u["cached"], output_tokens=u["output"], thinking_tokens=u["thinking"],
                           latency_ms=round(seconds * 1000), cost_usd=cost(model, **u)))
            db.commit()
    except Exception as e:
        print(f"   [observe] couldn't save a model call: {e!r}")


def install(c=client):
    """Wrap the shared client's two generate methods. Call once per process (API, worker, eval)."""
    models = c.models
    if getattr(models, "_metered", False):
        return
    ensure_tables()
    plain, plain_stream = models.generate_content, models.generate_content_stream

    def generate_content(*, model: str, **kw):
        t0 = time.perf_counter()
        response = plain(model=model, **kw)
        record(model, response.usage_metadata, time.perf_counter() - t0)
        return response

    def generate_content_stream(*, model: str, **kw):
        t0, last = time.perf_counter(), None
        for chunk in plain_stream(model=model, **kw):
            last = chunk
            yield chunk
        record(model, last.usage_metadata if last else None, time.perf_counter() - t0)  # usage is on the last chunk

    models.generate_content, models.generate_content_stream = generate_content, generate_content_stream
    models._metered = True


# --- The report --------------------------------------------------------------------------------------

PER_ROUTE = """
    WITH calls AS (
        SELECT run_id, COUNT(*) AS n, SUM(cost_usd) AS cost, SUM(prompt_tokens) AS prompt,
               SUM(output_tokens + thinking_tokens) AS output, BOOL_OR(cost_usd IS NULL) AS unpriced
        FROM llm_calls GROUP BY run_id)
    SELECT r.route, COUNT(*) AS runs,
           AVG(COALESCE(c.cost, 0)) AS avg_cost, MAX(COALESCE(c.cost, 0)) AS max_cost,
           AVG(COALESCE(c.n, 0)) AS avg_calls, AVG(COALESCE(c.prompt, 0)) AS avg_prompt,
           AVG(COALESCE(c.output, 0)) AS avg_output,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.seconds) AS p50,
           PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY r.seconds) AS p95,
           BOOL_OR(COALESCE(c.unpriced, false)) AS unpriced
    FROM agent_runs r LEFT JOIN calls c ON c.run_id = r.id
    WHERE r.created_at >= :since
    GROUP BY r.route ORDER BY avg_cost DESC"""

PER_MODEL = """
    SELECT model, COUNT(*) AS calls, SUM(cost_usd) AS cost, AVG(latency_ms) AS avg_ms,
           SUM(prompt_tokens) AS prompt, SUM(output_tokens) AS output, SUM(thinking_tokens) AS thinking
    FROM llm_calls WHERE created_at >= :since GROUP BY model ORDER BY cost DESC NULLS LAST"""


def rows(sql: str, since: str) -> list[dict]:
    with SessionLocal() as db:
        return [dict(r) for r in db.execute(text(sql), {"since": since}).mappings()]


def usd(x) -> str:
    return f"${float(x or 0):.4f}"


def report(since: str):
    routes = rows(PER_ROUTE, since)
    if not routes:
        print("   nothing recorded yet: run with --traffic, or use the app with the API/worker running")
        return
    header(f"Cost and latency per route (since {since})")
    print(f"   {'route':<20}{'runs':>5}{'avg cost':>11}{'max cost':>11}{'model calls':>12}"
          f"{'prompt tok':>11}{'out tok':>9}{'p50':>7}{'p95':>7}")
    for r in routes:
        print(f"   {r['route']:<20}{r['runs']:>5}{usd(r['avg_cost']):>11}{usd(r['max_cost']):>11}"
              f"{float(r['avg_calls']):>12.1f}{float(r['avg_prompt']):>11.0f}{float(r['avg_output']):>9.0f}"
              f"{r['p50']:>6.1f}s{r['p95']:>6.1f}s" + ("  (some calls unpriced)" if r["unpriced"] else ""))

    header("Per model")
    for m in rows(PER_MODEL, since):
        print(f"   {m['model']:<24}{m['calls']:>5} calls  {usd(m['cost']):>10}  avg {float(m['avg_ms']) / 1000:.1f}s  "
              f"tokens: {m['prompt']:,} in, {m['output']:,} out, {m['thinking']:,} thinking")

    by = {r["route"]: r for r in routes}
    if "investigation job" in by:
        dispatch = float(by.get("investigation", {}).get("avg_cost") or 0)
        job = by["investigation job"]
        total = dispatch + float(job["avg_cost"] or 0)
        header("So: what does an average investigation cost?")
        print(f"   {usd(total)} = {usd(dispatch)} routing (API) + {usd(job['avg_cost'])} the job (worker)")
        print(f"   {float(job['avg_calls']):.0f} model calls, p50 {job['p50']:.0f}s, p95 {job['p95']:.0f}s, "
              f"over {job['runs']} investigations")
        later = price_for("gemini-3.8-flash", date(2027, 1, 1))
        now = price_for("gemini-3.8-flash")
        if later and now and later != now:
            print(f"   from 01/01/2027 gemini-3.8-flash costs {later.input / now.input:.0f}x: "
                  "the main-model part of every number above doubles")
    quick = [r for r in routes if r["route"] not in ("investigation", "investigation job")]
    if quick:
        n = sum(r["runs"] for r in quick)
        avg = sum(float(r["avg_cost"] or 0) * r["runs"] for r in quick) / n
        print(f"\n   an average quick reply (smalltalk, clarify, simple reads; {n} runs): {usd(avg)}")
        print(f"   1,000 quick replies ~ {usd(avg * 1000)}; "
              + (f"1,000 investigations ~ {usd(total * 1000)}" if "investigation job" in by else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic", action="store_true", help="run 11.1's golden questions first (~5 min)")
    ap.add_argument("--since", default=f"{date.today():%Y-%m-%d}", help="report runs since this date (YYYY-MM-DD)")
    args = ap.parse_args()
    install()
    if args.traffic:
        from lessons import l111_evals
        sys.argv = ["l111_evals", "--no-judge"]  # the judge isn't part of the product: keep it out
        l111_evals.main()
    report(args.since)


if __name__ == "__main__":
    main()
