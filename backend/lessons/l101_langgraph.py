"""
Lesson 10.1: The orchestrator in LangGraph
===========================================

Run from backend/:   uv run python -m lessons.l101_langgraph

M7 was plain Python: a for loop over the plan's steps, a for loop for the retry. LangGraph writes
the same thing as a GRAPH: nodes (functions) that read a shared STATE and return changes to it,
and edges that say which node runs next.

    START -> planner -> worker -+-> worker (again, while steps are left)
                                +-> synthesizer -> validator -+-> END              (verified)
                                         ^                    +-> synthesizer      (one rewrite)
                                         +--------------------+
                                                              +-> safe_failure -> END

    state = {question, user_id, who, plan, evidence[], log[], report, problems[], attempt, reply, verified}

Every node is a thin wrapper around the SAME functions M7 uses (plan, run_worker, synthesize,
show_report, safe_failure). Only the control flow moved: from Python loops into the graph.

Reducers: evidence and log are "add" fields. A worker returns only ITS new evidence and calls;
LangGraph appends them to the state. Nobody mutates shared lists, so the state after every node
is a clean snapshot. That is what makes 10.2 (checkpoints: pause, save, resume) possible.

Result (23/11/2026): the 7.3 eval, same grading -> raw M7 3/3, LangGraph 3/3. Times within model noise
(80-151s per story, both versions).

What got EASIER with LangGraph
  - The flow is visible. The retry is an edge (validator -> synthesizer), not a for loop with a flag,
    and get_graph().draw_mermaid() draws it.
  - Progress for free: graph.stream(stream_mode="updates") yields after every node. In 8.2 we threaded a
    progress callback by hand through gather() and investigate().
  - Clean state snapshots (reducers, no shared mutable lists) -> checkpoints and pause/resume (10.2)
    come almost for free. Raw, we'd have to save and reload every variable ourselves.
  - Adding a step (an approval, a second validator) = one node + one edge, without touching the others.

What got HARDER
  - State is a loosely typed dict: a typo in a key is a runtime KeyError, not a wrong function argument.
  - Reducers are a trap: a worker that returned the WHOLE log would double it. It must copy the log,
    let the tools number new calls after it, and return only the new tail.
  - Errors and stack traces go through LangGraph's internals; prints inside nodes are still how you debug.
  - More lines for the same flow (nodes + edge functions + state ~90 lines vs ~45 lines of loops),
    a new dependency with fast-moving APIs, and a recursion limit (25 node runs by default).
  - It replaced only the LOOPS. Everything hard stayed ours: the plan check, step-scoped tools, the
    shared numbered log, citations, re-checking facts against the DB, safe failure.

Experiments:
  1. The graph itself (Mermaid text: paste into mermaid.live to see it)
  2. One investigation, streamed node by node (progress for free)
  3. The 7.3 eval: the three planted stories, raw M7 vs the graph, same grading
"""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from lessons.common import header
from lessons.l21_permissions import load_user
from lessons.l47_isolation import who
from lessons.l62_verify import numbered
from lessons.l63_rules import Context, safe_failure
from lessons.l71_planner import Plan, check_plan, plan
from lessons.l72_workers import Evidence, Report, run_worker, show_report, synthesize, worker_tools
from lessons.l73_investigation import EXTRA_RULES, RETRIES, grade, stories, with_sources
from lessons.l73_investigation import investigate as investigate_raw

ASHA = 1


class State(TypedDict, total=False):
    question: str
    user_id: int
    who: str
    plan: Plan
    evidence: Annotated[list[Evidence], operator.add]  # each worker returns its new item; the graph appends
    log: Annotated[list, operator.add]                 # every tool call (name, args, result), call_id = position
    report: Report
    problems: list[str]
    attempt: int
    reply: str
    verified: bool


# --- Nodes: thin wrappers around M7's functions -------------------------------------------------

def planner(s: State) -> dict:
    p = plan(s["question"])
    if problems := check_plan(p):
        print(f"   plan problems (continuing anyway): {problems}")
    return {"plan": p, "who": who(load_user(s["user_id"])), "attempt": 0}


def worker(s: State) -> dict:
    step = s["plan"].steps[len(s["evidence"])]  # the next step not done yet
    log = list(s["log"])                        # a copy, so call_ids continue from the shared log
    start = len(log)
    finding = run_worker(step, s["evidence"], numbered(worker_tools(s["user_id"]), log), s["who"])
    return {"evidence": [Evidence(step.step, step.goal, finding, list(range(start + 1, len(log) + 1)))],
            "log": log[start:]}                 # only what's new: the reducer appends it


def synthesizer(s: State) -> dict:
    feedback = ""
    if s.get("problems"):  # second time round: the validator's problems go back in
        feedback = ("\n\nYour previous report was REJECTED by automatic checks:\n"
                    + "\n".join(f"- {p}" for p in s["problems"])
                    + "\nWrite it again from the same evidence. Fix or drop every claim you can't cite exactly.")
    report = synthesize(s["question"], s["evidence"], s["log"], s["who"], feedback)
    return {"report": report, "attempt": s["attempt"] + 1}


def validator(s: State) -> dict:
    clean = worker_tools(s["user_id"])
    c = Context(s["question"], load_user(s["user_id"]), clean)
    problems = show_report(s["report"], s["evidence"], s["log"], clean)
    problems += [f"{name}: {p}" for name, rule in EXTRA_RULES.items() for p in rule(s["report"], s["log"], c)]
    if problems:
        print(f"\n   attempt {s['attempt']}: REJECTED")
        for p in problems:
            print(f"     XX {p}")
        return {"problems": problems, "verified": False}
    return {"problems": [], "verified": True, "reply": with_sources(s["report"], s["log"])}


def give_up(s: State) -> dict:
    c = Context(s["question"], load_user(s["user_id"]), worker_tools(s["user_id"]))
    return {"reply": safe_failure(s["report"], s["log"], c)}


# --- Edges: the control flow that used to be Python loops --------------------------------------

def after_worker(s: State) -> str:
    return "worker" if len(s["evidence"]) < len(s["plan"].steps) else "synthesizer"


def after_validator(s: State) -> str:
    if s["verified"]:
        return END
    return "synthesizer" if s["attempt"] <= RETRIES else "give_up"


def build_graph(approve=None) -> StateGraph:
    """-> the graph, not compiled yet (10.2 compiles it with a checkpointer).
    approve: 10.2's gate between planner and workers; it sets reply to stop early."""
    g = StateGraph(State)
    for name, fn in [("planner", planner), ("worker", worker), ("synthesizer", synthesizer),
                     ("validator", validator), ("give_up", give_up)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    if approve:
        g.add_node("approve", approve)
        g.add_edge("planner", "approve")
        g.add_conditional_edges("approve", lambda s: END if s.get("reply") else "worker", [END, "worker"])
    else:
        g.add_edge("planner", "worker")
    g.add_conditional_edges("worker", after_worker, ["worker", "synthesizer"])
    g.add_edge("synthesizer", "validator")
    g.add_conditional_edges("validator", after_validator, [END, "synthesizer", "give_up"])
    g.add_edge("give_up", END)
    return g


GRAPH = build_graph().compile()


def investigate(question: str, user_id: int, show_nodes: bool = False) -> tuple[str, bool]:
    """Same contract as 7.3's investigate: -> (reply, verified?)."""
    state: State = {"question": question, "user_id": user_id, "evidence": [], "log": []}
    if not show_nodes:
        final = GRAPH.invoke(state)
        return final["reply"], final["verified"]
    t0, final = time.perf_counter(), dict(state)
    for update in GRAPH.stream(state, stream_mode="updates"):  # one item per node that finished
        for node, changes in update.items():
            final.update({k: final.get(k, []) + v if k in ("evidence", "log") else v
                          for k, v in (changes or {}).items()})
            print(f"   >>> [{time.perf_counter() - t0:5.1f}s] node '{node}' done: {sorted((changes or {}).keys())}")
    return final["reply"], final["verified"]


# --- Experiments ---------------------------------------------------------------------------------

def main():
    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: the graph")
    print(GRAPH.get_graph().draw_mermaid())

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: one investigation, node by node")
    reply, ok = investigate("Why are we losing money on padlocks in BLR?", ASHA, show_nodes=True)
    print(f"\n   verified={ok}\n{reply[:400]}...")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: the 7.3 eval, raw M7 vs LangGraph")
    rows = []
    for story in stories():
        for name, run in (("raw M7", investigate_raw), ("LangGraph", investigate)):
            t0 = time.perf_counter()
            reply, _ = run(story.question, ASHA)
            rows.append((story.name, name, grade(story, reply, "investigation"), time.perf_counter() - t0))

    header("Experiment 3: scorecard (truth from SQL, 7.3's grade)")
    for story, name, missing, secs in rows:
        print(f"   {'FOUND ' if not missing else 'MISSED'}  {story:<22} {name:<10} {secs:5.0f}s"
              + "".join(f"\n            XX {m}" for m in missing))
    for name in ("raw M7", "LangGraph"):
        found = sum(not m for _, n, m, _ in rows if n == name)
        print(f"\n   {name}: {found}/3 stories found with correct numbers")


if __name__ == "__main__":
    main()
