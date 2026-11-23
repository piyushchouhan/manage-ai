"""
Lesson 10.2: Checkpoints and human-in-the-loop
===============================================

Run from backend/:   uv run python -m lessons.l102_checkpoints
Resume by hand:      uv run python -m lessons.l102_checkpoints resume <thread_id> yes|no

Some steps should not run without a person saying yes: scanning every open invoice, and later
anything that writes, sends or costs real money. So the graph PAUSES before them and asks.

    START -> planner -> APPROVE -> worker ... -> END
                           |
                           +-- the plan uses a costly tool?  interrupt("Step 1 will scan every open
                               invoice. Continue?")  -> the graph STOPS here

A pause can last minutes or days, and the process that started it may be long gone. So after EVERY
node the state is saved: a CHECKPOINT, in Postgres (tables checkpoints, checkpoint_blobs, ...),
under a thread_id. Resuming = load the last checkpoint of that thread and carry on:

    graph.invoke(state, {"configurable": {"thread_id": "inv-1a2b"}})   -> pauses, returns __interrupt__
    ... any time later, any process ...
    graph.invoke(Command(resume="yes"), same thread)                  -> approve gets "yes", workers run

The planner does NOT run again on resume: its result is in the checkpoint. Only the node that paused
(approve) runs again, from its start, with interrupt() now returning the answer. So everything in a
node BEFORE interrupt() must be safe to repeat (here: it only reads the plan).

Experiments:
  1. "Which customers owe us money?" -> the plan uses list_unpaid_invoices -> paused. What's saved.
  2. Resume with "yes" in a NEW process -> workers, answer. Planner not re-run.
  3. The same question, answered "no"  -> stops, no tool runs
  4. "Why are we losing money on padlocks in BLR?" -> no costly tool, no pause
"""

import subprocess
import sys
import time
import uuid
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command, interrupt
from psycopg import Connection
from psycopg.rows import dict_row

from app.config import DATABASE_URL
from lessons.common import header
from lessons.l101_langgraph import State, build_graph
from lessons.l62_verify import Fact
from lessons.l71_planner import Plan, Step
from lessons.l72_workers import CitedFact, Evidence, Report

ASHA = 1
CHECKPOINT_DB = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")  # psycopg's own URL form
COSTLY = {"list_unpaid_invoices": "scan every open invoice in your warehouses"}  # tool -> what to tell the user
YES = {"yes", "y", "ok", "continue", "go", "haan", "ha"}

# Loading a checkpoint rebuilds Python objects from the DB. Only these classes may be rebuilt: anyone who
# can write to the checkpoint tables must not be able to make us construct something arbitrary.
SERDE = JsonPlusSerializer(allowed_msgpack_modules=[Plan, Step, Evidence, Report, CitedFact, Fact])


# --- The gate --------------------------------------------------------------------------------------

def approve(s: State) -> dict:
    """Runs twice when it pauses: once until interrupt(), again on resume. Nothing here may have effects."""
    costly = [(st.step, t) for st in s["plan"].steps for t in st.suggested_tools if t in COSTLY]
    if not costly:
        return {}
    ask = " ".join(f"Step {n} will {COSTLY[t]} ({t})." for n, t in costly) + " Continue?"
    answer = interrupt({"ask": ask, "plan": [f"{st.step}. {st.goal}" for st in s["plan"].steps]})
    if str(answer).strip().lower() in YES:
        return {}
    return {"reply": "Okay, I stopped before running it. Nothing was scanned.", "verified": False}


@contextmanager
def checkpointed():
    """The 10.1 graph + the gate, saving a checkpoint after every node."""
    with Connection.connect(CHECKPOINT_DB, autocommit=True, prepare_threshold=0, row_factory=dict_row) as conn:
        saver = PostgresSaver(conn, serde=SERDE)  # = PostgresSaver.from_conn_string, plus our allow-list
        saver.setup()  # creates the checkpoint tables the first time
        yield build_graph(approve).compile(checkpointer=saver)


def thread(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def run(graph, start, thread_id: str) -> dict:
    """Run until the end or a pause, printing each node as it finishes. -> the final values (or the pause)."""
    t0, out = time.perf_counter(), {}
    for update in graph.stream(start, thread(thread_id), stream_mode="updates"):
        for node, changes in update.items():
            if node == "__interrupt__":
                out["paused"] = changes[0].value
                print(f"   >>> [{time.perf_counter() - t0:5.1f}s] PAUSED: {changes[0].value['ask']}")
            else:
                print(f"   >>> [{time.perf_counter() - t0:5.1f}s] node '{node}' done")
    if "paused" not in out:
        values = graph.get_state(thread(thread_id)).values
        out.update(reply=values.get("reply"), verified=values.get("verified"))
    return out


def start(question: str, user_id: int) -> tuple[str, dict]:
    thread_id = f"inv-{uuid.uuid4().hex[:8]}"
    with checkpointed() as g:
        return thread_id, run(g, {"question": question, "user_id": user_id, "evidence": [], "log": []}, thread_id)


def resume(thread_id: str, answer: str) -> dict:
    with checkpointed() as g:
        return run(g, Command(resume=answer), thread_id)


def saved(thread_id: str):
    with checkpointed() as g:
        st = g.get_state(thread(thread_id))
        history = list(g.get_state_history(thread(thread_id)))
    print(f"   saved in Postgres for thread {thread_id}:")
    print(f"     checkpoints: {len(history)}   next node to run: {st.next}")
    print(f"     plan in the state: {[f'{s.step}. {s.goal}' for s in st.values['plan'].steps]}")
    print(f"     tool calls so far: {len(st.values.get('log', []))}")


# --- Experiments ---------------------------------------------------------------------------------

def resume_in_new_process(thread_id: str, answer: str):
    """A different Python process: nothing in memory, only the checkpoint in Postgres."""
    print(f"   $ python -m lessons.l102_checkpoints resume {thread_id} {answer}   (new process)")
    subprocess.run([sys.executable, "-u", "-m", "lessons.l102_checkpoints", "resume", thread_id, answer], check=True)


def main():
    q = "Which customers owe us money, and how much?"

    # --- Experiment 1 ---------------------------------------------------------
    header(f"Experiment 1: '{q}'")
    tid, out = start(q, ASHA)
    print(f"\n   the user is asked: {out.get('paused', {}).get('ask')}")
    saved(tid)

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: resume with 'yes', in a new process")
    resume_in_new_process(tid, "yes")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: the same question, answered 'no'")
    tid2, out = start(q, ASHA)
    resume_in_new_process(tid2, "no")

    # --- Experiment 4 ---------------------------------------------------------
    header("Experiment 4: 'Why are we losing money on padlocks in BLR?' (no costly tool)")
    _, out = start("Why are we losing money on padlocks in BLR?", ASHA)
    print(f"\n   paused: {'paused' in out}   verified: {out.get('verified')}\n{(out.get('reply') or '')[:300]}...")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "resume":
        result = resume(sys.argv[2], sys.argv[3])
        print(f"\n   verified: {result.get('verified')}\n   reply:\n{(result.get('reply') or '')[:500]}")
    else:
        main()
