"""
Lesson 11.1: End-to-end evals
==============================

Run from backend/ BEFORE EVERY CHANGE:   uv run python -m lessons.l111_evals
Options:   --only stock-01,story     (ids starting with these)   --no-judge   --workers 4

Until now each lesson tested its own piece. This tests the WHOLE assistant the way a user meets it:
the same entrance as the API (router -> quick path with tables, or investigation job), 25 golden
questions in evals/golden.jsonl.

Each question is scored twice:

  1. CODE (deterministic, cheap, never flaky)        2. JUDGE (an LLM, second opinion)
     - took the expected route?                         reads the question, the true facts and what the
     - shows every expected fact? (truth = SQL now,      user saw; scores 1-5. Catches what code can't:
       not hard-coded: survives a re-seed)               "answered a different question", "unclear",
     - shows nothing it must not? (other users'          "picked a box instead of asking which".
       warehouses, a MUM invoice to Ravi)
     - required words? ("?" when it should ask back)    Its score is an opinion: it never overrides
     - no numbers in small talk?                         a code check, only adds to it.

What the user SAW is graded: the reply AND the tables/metrics under it (9.2).

Every run is saved to evals/results/<time>.json and compared with the previous run, so a change that
breaks something shows up as a REGRESSION, by name.
"""

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from app.db import query
from app.llm import MODEL, client, types
from lessons.l21_permissions import load_user
from lessons.l41_history import load_messages, new_session
from lessons.l62_verify import numbers_in, same
from lessons.l81_jobs import get_job, start_turn
from lessons.l92_rendering import HANDLERS_V3, ensure_tables
from lessons.l112_observability import install

GOLDEN = Path("evals/golden.jsonl")
RESULTS = Path("evals/results")
FAILED_ANSWER = ("I couldn't produce", "Sorry, the investigation failed", "Sorry, the server restarted")

STOCK_SQL = """SELECT s.quantity FROM stock_levels s JOIN products p ON p.id = s.product_id
               JOIN warehouses w ON w.id = s.warehouse_id WHERE p.sku = :sku AND w.code = :wh"""
LOW_SQL = """SELECT p.sku FROM stock_levels s JOIN products p ON p.id = s.product_id
             JOIN warehouses w ON w.id = s.warehouse_id WHERE s.quantity < p.reorder_level AND w.code = :wh"""


# --- The truth, from SQL at eval time --------------------------------------------------------------

def truth(spec: dict, user_warehouses: set[str]) -> list:
    """A fact spec from golden.jsonl -> the values the user must (or must not) see."""
    if "stock" in spec:
        sku, wh = spec["stock"]
        rows = query(STOCK_SQL, sku=sku, wh=wh)
    elif "low" in spec:
        rows = query(LOW_SQL, wh=spec["low"])
    elif "low_only" in spec:  # low there, and NOT low anywhere this user may see (those they may be shown)
        theirs = {r["sku"] for wh in user_warehouses for r in query(LOW_SQL, wh=wh)}
        rows = [r for r in query(LOW_SQL, wh=spec["low_only"]) if r["sku"] not in theirs]
    else:
        rows = query(spec["sql"])
    return [next(iter(r.values())) for r in rows]


def shown(value, text: str, numbers: list[float]) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return any(same(value, n) for n in numbers)
    return str(value).lower() in text.lower()


# --- One question, through the real entrance --------------------------------------------------------

class Inline:
    """Like 8.2's queue, but runs the investigation right here, so the eval can wait for it."""
    def add_task(self, fn, *args):
        fn(*args)


def ask(item: dict) -> dict:
    user = item.get("user", 1)
    sid = new_session(user)
    for earlier in item.get("earlier", []):
        start_turn(user, sid, earlier, Inline(), handlers=HANDLERS_V3)
    t0 = time.perf_counter()
    out = start_turn(user, sid, item["question"], Inline(), handlers=HANDLERS_V3)
    reply = out["reply"] if out.get("reply") is not None else (get_job(user, out["job_id"]) or {}).get("reply") or ""
    display = load_messages(sid)[-1].get("display") or {}
    return {"path": out["path"], "reply": reply, "blocks": display.get("blocks", []),
            "seconds": round(time.perf_counter() - t0, 1)}


# --- 1. Code checks ------------------------------------------------------------------------------

def code_checks(item: dict, run: dict, user_warehouses: set[str]) -> list[str]:
    seen = run["reply"] + "\n" + json.dumps(run["blocks"], default=str)  # what the user saw
    numbers = numbers_in(seen)
    problems = []
    if run["path"] not in item["route"]:
        problems.append(f"route {run['path']}, expected {' or '.join(item['route'])}")
    for spec in item.get("facts", []):
        alternatives = spec.get("or_text", [])
        for v in truth(spec, user_warehouses):
            if not shown(v, seen, numbers) and not any(t.lower() in seen.lower() for t in alternatives):
                problems.append(f"doesn't show {v}")
    for spec in item.get("not", []):
        problems += [f"shows {v}, which it must not" for v in truth(spec, user_warehouses) if shown(v, seen, numbers)]
    for group in item.get("mention", []):
        if not any(w.lower() in seen.lower() for w in group):
            problems.append(f"doesn't say any of {group}")
    if item.get("no_numbers") and numbers_in(run["reply"]):
        problems.append(f"states numbers in small talk: {numbers_in(run['reply'])}")
    if run["reply"].startswith(FAILED_ANSWER):
        problems.append("safe failure / job failed")
    return problems


# --- 2. The judge -----------------------------------------------------------------------------------

class Judgement(BaseModel):
    reason: str = Field(description="One or two sentences: what is good or wrong.")
    score: int = Field(ge=1, le=5)


JUDGE_PROMPT = (
    "You grade one answer of a read-only warehouse assistant. You get the question, who asked and which "
    "warehouses they may see, the correct facts from the database, and exactly what the user saw (reply text, "
    "plus tables shown under it as JSON). Score 1-5: 5 = answers what was asked, consistent with the facts, "
    "nothing invented, nothing from warehouses the user may not see, clear. 4 = right with small issues. "
    "3 = partly right or unclear. 2 = mostly wrong. 1 = wrong, invented, or leaked data. Facts in tables count "
    "as shown. Be strict about invented numbers and leaks, lenient about wording."
)


def judge(item: dict, run: dict, user, expected: dict) -> Judgement:
    content = (f"Question: {item['question']}\n"
               + (f"Earlier in the chat: {item['earlier']}\n" if item.get("earlier") else "")
               + f"Asked by: {user.name} ({user.role}), may see warehouses {sorted(user.warehouse_codes)}\n"
               f"Correct facts (from the database): {expected['facts'] or 'none needed'}\n"
               f"Must NOT be shown: {expected['not'] or 'nothing specific'}\n"
               f"Grading note: {item.get('rubric', 'none')}\n\n"
               f"What the user saw:\n{run['reply']}\n\nTables under it: {json.dumps(run['blocks'], default=str)[:3000]}")
    r = client.models.generate_content(
        model=MODEL, contents=content,
        config=types.GenerateContentConfig(system_instruction=JUDGE_PROMPT, temperature=0,
                                           response_mime_type="application/json", response_schema=Judgement))
    return Judgement.model_validate_json(r.text)


def evaluate(item: dict, use_judge: bool) -> dict:
    user = load_user(item.get("user", 1))
    try:
        run = ask(item)
    except Exception as e:  # a crash is a failed question, not a crashed eval
        return {"id": item["id"], "path": "-", "reply": "", "seconds": 0, "code": [f"crashed: {e!r}"],
                "score": None, "reason": ""}
    whs = set(user.warehouse_codes)
    expected = {"facts": [v for s in item.get("facts", []) for v in truth(s, whs)],
                "not": [v for s in item.get("not", []) for v in truth(s, whs)]}
    result = {"id": item["id"], "path": run["path"], "reply": run["reply"], "seconds": run["seconds"],
              "code": code_checks(item, run, whs), "score": None, "reason": ""}
    if use_judge:
        j = judge(item, run, user, expected)
        result.update(score=j.score, reason=j.reason)
    return result


# --- The scorecard ----------------------------------------------------------------------------------

def passed(r: dict) -> bool:
    return not r["code"] and (r["score"] is None or r["score"] >= 4)


def previous_run(before: Path) -> tuple[str, dict] | None:
    runs = sorted(p for p in RESULTS.glob("*.json") if p != before)
    if not runs:
        return None
    return runs[-1].stem, {r["id"]: r for r in json.loads(runs[-1].read_text())}


def scorecard(results: list[dict], wall: float, saved: Path):
    print(f"\n{'question':<26}{'route':<15}{'code':<6}{'judge':<7}{'time':>6}")
    for r in results:
        print(f"{r['id']:<26}{r['path']:<15}{'PASS' if not r['code'] else 'FAIL':<6}"
              f"{r['score'] if r['score'] is not None else '-':<7}{r['seconds']:>5.0f}s")
        for p in r["code"]:
            print(f"{'':<4}XX {p}")
        if r["score"] is not None and r["score"] < 4:
            print(f"{'':<4}judge: {r['reason']}")

    print("\nby group:")
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["id"].split("-")[0], []).append(r)
    for g, rs in groups.items():
        print(f"   {g:<8}{sum(passed(r) for r in rs)}/{len(rs)}")

    n = len(results)
    scores = [r["score"] for r in results if r["score"] is not None]
    print(f"\nTOTAL   {sum(passed(r) for r in results)}/{n} passed   "
          f"code checks {sum(not r['code'] for r in results)}/{n}"
          + (f"   judge avg {sum(scores) / len(scores):.1f}/5" if scores else "")
          + f"   wall {wall / 60:.1f} min   saved {saved}")

    if prev := previous_run(saved):
        name, before = prev
        when = datetime.strptime(name, "%Y%m%d-%H%M%S").strftime("%d/%m/%Y %H:%M")
        worse = [r["id"] for r in results if r["id"] in before and passed(before[r["id"]]) and not passed(r)]
        better = [r["id"] for r in results if r["id"] in before and not passed(before[r["id"]]) and passed(r)]
        print(f"vs the run of {when}:   {len(worse)} REGRESSION(S) {worse or ''}   {len(better)} fixed {better or ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated id prefixes")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ensure_tables()
    install()  # 11.2: the eval's calls are measured too (the judge's outside any run)
    items = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    if args.only:
        items = [i for i in items if i["id"].startswith(tuple(args.only.split(",")))]
    print(f"{len(items)} golden questions, {args.workers} at a time, judge {'off' if args.no_judge else MODEL}")

    out, t0, results = sys.stdout, time.perf_counter(), []
    sys.stdout = io.StringIO()  # the agent's step-by-step prints, from 4 threads at once: not readable
    try:
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(evaluate, item, not args.no_judge): item for item in items}
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                print(f"  [{len(results):>2}/{len(items)}] {'PASS' if passed(r) else 'FAIL'}  {r['id']}  "
                      f"({r['seconds']:.0f}s)", file=out, flush=True)
    finally:
        sys.stdout = out

    order = [i["id"] for i in items]
    results.sort(key=lambda r: order.index(r["id"]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    saved = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    saved.write_text(json.dumps(results, indent=1, default=str))
    scorecard(results, time.perf_counter() - t0, saved)


if __name__ == "__main__":
    main()
