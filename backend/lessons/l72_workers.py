"""
Lesson 7.2: Workers and synthesizer
====================================

Run from backend/:   uv run python -m lessons.l72_workers

7.1's planner wrote the to-do list. Now somebody does the work, and somebody writes it up:

    plan = planner(question)                      7.1: 1 call, no tools
    for step in plan.steps:                       CODE runs the steps, in order
        evidence += worker(step, evidence so far)     a small agent loop (M1) with ONLY that
                                                      step's tools, a 4-step budget, and the
                                                      earlier findings (step 2 learns the SKU here)
    report = synthesizer(question, evidence)      1 call, NO tools: writes from the evidence only
    check(report)                                 6.2: every fact re-checked against the DB

Every tool call of every worker goes into ONE shared log, so each result has a unique
call_id. The synthesizer must cite every number as {step, call_id, field}. Code then checks:
    - the call really happened in THAT step     (a citation can't be made up)
    - the value matches the DB                  (6.2's check_fact)
    - no number in the summary without a fact   (6.2's unbacked)

Why split it this way?
    worker       small context: one goal, 1-2 tools, the earlier findings. Not the whole history.
    synthesizer  sees only evidence, no tools: it can't go fetch more or wander, only write.

Experiments:
  1. The padlock question (shrinkage story)
  2. The customers question (non-paying customers story), with 7.2's new list_unpaid_invoices
"""

import json
import time
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, Field

from app.llm import MODEL, client, types
from lessons.common import header
from lessons.l12_agent_loop import TooManySteps, run_agent
from lessons.l21_permissions import load_user
from lessons.l22_tool_design import safe
from lessons.l46_memory import search_notes
from lessons.l47_isolation import scope_for, who
from lessons.l61_typed_tools import TOOLS_V7
from lessons.l62_verify import Answer, Fact, check_fact, numbered, unbacked
from lessons.l71_planner import WORKER_DECLS, Plan, Step, check_plan, plan

ASHA = 1
WORKER_STEPS = 4  # step budget per worker: a step is ONE piece of evidence, not an investigation


@dataclass
class Evidence:
    step: int
    goal: str
    finding: str
    calls: list[int] = field(default_factory=list)  # call_ids in the shared log


# --- Workers --------------------------------------------------------------------------------

def worker_tools(user_id: int) -> dict:
    """Every worker tool, scoped to the user (4.7). Each worker gets only its step's subset."""
    scope = scope_for(load_user(user_id))
    raw = {**{n: fn for n, fn in TOOLS_V7.items() if n in WORKER_DECLS}, "search_notes": safe(search_notes)}
    return {n: scope(fn) for n, fn in raw.items()}


WORKER_PROMPT = (
    "You do ONE step of an investigation for a warehouse assistant. Today is {today}.{who}\n\n"
    "Your step: {goal}\n\nWhat earlier steps found:\n{earlier}\n\n"
    "Use your tools to get exactly this evidence. Then reply in 2-4 sentences: what you found, with exact "
    "numbers, SKUs, names and dates (DD/MM/YYYY). Do not answer the user's overall question. Do not guess."
)


def run_worker(step: Step, done: list[Evidence], tools: dict, who_text: str) -> str:
    earlier = "\n".join(f"- step {e.step} ({e.goal}): {e.finding}" for e in done) or "(nothing yet: this is step 1)"
    config = types.GenerateContentConfig(
        system_instruction=WORKER_PROMPT.format(today=f"{date.today():%d/%m/%Y}", who=who_text,
                                                goal=step.goal, earlier=earlier),
        tools=[types.Tool(function_declarations=[WORKER_DECLS[n] for n in step.suggested_tools])],
        temperature=0,
    )
    only_mine = {n: tools[n] for n in step.suggested_tools}  # code enforces it, not just the prompt
    try:
        return run_agent(f"Do step {step.step}: {step.goal}", config=config, registry=only_mine,
                         max_steps=WORKER_STEPS)
    except TooManySteps:
        return "This step ran out of its step budget before finishing."


# --- Synthesizer --------------------------------------------------------------------------------

class CitedFact(Fact):
    step: int = Field(description="The step whose evidence this value comes from.")


class Report(Answer):  # 6.2's answer shape, with each fact also citing its step
    summary: str = Field(description="The answer to the question. Cite steps inline like [step 2]. Dates DD/MM/YYYY.")
    facts: list[CitedFact] = Field(description="One entry for EVERY number in the summary.")  # pyrefly: ignore[bad-override]


SYNTH_PROMPT = (
    "You write the final answer to an investigation, using ONLY the evidence given. You have no tools. "
    "Explain the cause: say what is unusual and compare it with its limit or with the others (owed vs credit "
    "limit, never paid, lead time vs other suppliers, unexplained vs normal movements). "
    "Every number in the summary must also be a fact, copied exactly from a tool result, with its step, call_id "
    "and the field path inside that result (keys and list positions joined by dots, e.g. 'customers.0.outstanding'). "
    "Do not calculate new numbers. If evidence is missing or a step failed, say so; never fill the gap.{who}"
)


def evidence_text(evidence: list[Evidence], log: list) -> str:
    parts = []
    for e in evidence:
        lines = [f"STEP {e.step}: {e.goal}", f"worker's finding: {e.finding}", "tool results:"]
        for cid in e.calls:
            name, args, result = log[cid - 1]
            lines.append(f"  [call_id {cid}] {name}({json.dumps(dict(args))}) -> {json.dumps(result)[:4000]}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def synthesize(question: str, evidence: list[Evidence], log: list, who_text: str, feedback: str = "") -> Report:
    """feedback: 7.3's validator sends the rejected report's problems back for one rewrite."""
    r = client.models.generate_content(
        model=MODEL, contents=f"Question: {question}\n\n{evidence_text(evidence, log)}{feedback}",
        config=types.GenerateContentConfig(
            system_instruction=SYNTH_PROMPT.format(who=who_text), temperature=0,
            response_mime_type="application/json", response_schema=Report,
        ),
    )
    return Report.model_validate_json(r.text)


# --- Checks: code only ------------------------------------------------------------------------------

def check_report(report: Report, evidence: list[Evidence], log: list, clean: dict) -> list[tuple[CitedFact, bool, str]]:
    step_of = {cid: e.step for e in evidence for cid in e.calls}
    out = []
    for f in report.facts:
        if f.call_id not in step_of:
            out.append((f, False, f"call #{f.call_id} never happened"))
        elif step_of[f.call_id] != f.step:
            out.append((f, False, f"cites step {f.step}, but call #{f.call_id} was made in step {step_of[f.call_id]}"))
        else:
            out.append((f, *check_fact(f, log, clean)))
    return out


def show_report(report: Report, evidence: list[Evidence], log: list, clean: dict) -> list[str]:
    """Print the answer and its sources. -> the problems (empty = VERIFIED)."""
    checks = check_report(report, evidence, log, clean)
    extra = unbacked(report)
    print(f"\n   ANSWER: {report.summary}\n\n   SOURCES:")
    for f, good, detail in checks:
        tool = log[f.call_id - 1][0] if 1 <= f.call_id <= len(log) else "?"
        print(f"     {'ok' if good else 'XX'}  [step {f.step} · {tool} · call #{f.call_id} · {f.field}] "
              f"{f.claim} = {f.value}  ({detail})")
    if extra:
        print(f"     XX  numbers in the summary with no fact: {extra}")
    return ([f"{f.claim} = {f.value}: {detail}" for f, good, detail in checks if not good]
            + [f"the summary states {n:g}, which is not a cited fact" for n in extra])


# --- The orchestrator ---------------------------------------------------------------------------------

@dataclass
class Investigation:
    plan: Plan
    evidence: list[Evidence]
    log: list    # every tool call of every worker, call_id = position + 1
    clean: dict  # the same scoped tools, NOT numbered: for re-checking against the DB
    who_text: str


def gather(question: str, user_id: int) -> Investigation:
    """Plan, then run one worker per step. Everything except writing the answer."""
    who_text = who(load_user(user_id))
    p: Plan = plan(question)
    print(f"\n   PLAN: {p.approach}")
    for s in p.steps:
        print(f"     {s.step}. {s.goal}  [{', '.join(s.suggested_tools)}]")
    if problems := check_plan(p):
        print(f"   plan problems (continuing anyway): {problems}")

    clean = worker_tools(user_id)
    log: list = []
    tools = numbered(clean, log)   # one shared log: call_ids unique across all workers
    evidence: list[Evidence] = []
    for s in p.steps:
        start = len(log)
        finding = run_worker(s, evidence, tools, who_text)
        evidence.append(Evidence(s.step, s.goal, finding, list(range(start + 1, len(log) + 1))))
        print(f"\n   STEP {s.step} done, calls {evidence[-1].calls}: {finding}")
    return Investigation(p, evidence, log, clean, who_text)


def investigate(question: str, user_id: int) -> tuple[Report, bool]:
    t0 = time.perf_counter()
    inv = gather(question, user_id)
    report = synthesize(question, inv.evidence, inv.log, inv.who_text)
    ok = not show_report(report, inv.evidence, inv.log, inv.clean)
    print(f"\n   {'VERIFIED' if ok else 'NOT VERIFIED'} · {len(inv.plan.steps)} steps · {len(inv.log)} tool calls · "
          f"{time.perf_counter() - t0:.0f}s")
    return report, ok


def main():
    header("Experiment 1: the padlock question (shrinkage)")
    investigate("Why are we losing money on padlocks in BLR?", ASHA)

    header("Experiment 2: the customers question (non-paying customers)")
    investigate("Which customers are hurting our cash flow, and how much do they owe?", ASHA)


if __name__ == "__main__":
    main()
