"""
Lesson 7.1: Planner
====================

Run from backend/:   uv run python -m lessons.l71_planner

A "why" question can't be answered with one lookup. "Why are we losing money on padlocks
in BLR?" needs: which product, how stock moved, which movements have no reason, what the
staff notes say, what was sold. The agent loop (M1) CAN wander through that, but:
    - nobody can see or check its plan before it spends money
    - it forgets parts, or goes round in circles (6.3's "ran out of steps")

Orchestration splits the job, like a manager and a team:

    PLANNER      one structured-output call, NO tools: question -> 3-5 steps     (this lesson)
    WORKERS      each step = a small agent loop with only its tools              (7.2)
    SYNTHESIZER  writes the answer from the evidence, citing each step          (7.2)

The planner only DECIDES the steps; it never touches data. Its output is data too, so code
checks it before anything runs (M6's idea, applied to plans):
    max 5 steps                 the schema says so
    only tools that exist       the tool name is a Literal built from the real tool list
    steps numbered 1..n         code renumbers, it doesn't trust the model
    no "write the answer" step  that's the synthesizer's job, not a worker's

Experiments:
  1. The padlock question -> a plan
  2. The other two planted stories (customers who don't pay, a slow supplier)
  3. A huge question: the 5-step cap forces the planner to choose
  4. A simple question: the planner is overkill here (that's why 5.2's router exists)
"""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from app.llm import MODEL, client, types
from lessons.common import header
from lessons.l46_memory import SEARCH_NOTES_DECL
from lessons.l61_typed_tools import DECLS_V7

MAX_STEPS = 5
WORKER_DECLS = {**{n: d for n, d in DECLS_V7.items() if n != "ask_user"}, "search_notes": SEARCH_NOTES_DECL}
# pyrefly: ignore[invalid-literal]  (built at runtime from the real tool list: fine for Pydantic)
ToolName = Literal[tuple(WORKER_DECLS)]  # the planner can only name tools that exist


class Step(BaseModel):
    step: int
    goal: str = Field(description="The ONE piece of evidence this step gets, e.g. 'unexplained adjustments for "
                                  "HAR-0018 in BLR over the last 3 months'.")
    suggested_tools: list[ToolName] = Field(min_length=1, description="Tools a worker would use for this step.")


class Plan(BaseModel):
    approach: str = Field(description="One sentence: how the steps together answer the question.")
    steps: list[Step] = Field(min_length=1, max_length=MAX_STEPS)


def tool_catalogue() -> str:
    return "\n".join(f"- {n}: {d.description}" for n, d in WORKER_DECLS.items())


PLANNER_PROMPT = (
    "You plan investigations for a warehouse assistant (warehouses BLR, MUM, DEL). Today is {today}.\n"
    "Break the question into at most {max} steps. Each step gathers ONE kind of evidence with the tools below. "
    "Steps run in order; a later step can use what earlier steps found (e.g. a SKU). "
    "Use as FEW steps as the question needs. Do not answer the question. Do not add a step that writes, "
    "summarizes or explains the answer: a separate writer does that from the evidence.\n\n"
    "Tools the workers have:\n{tools}"
)


def plan(question: str) -> Plan:
    r = client.models.generate_content(
        model=MODEL, contents=question,
        config=types.GenerateContentConfig(
            system_instruction=PLANNER_PROMPT.format(today=f"{date.today():%d/%m/%Y}", max=MAX_STEPS,
                                                     tools=tool_catalogue()),
            temperature=0, response_mime_type="application/json", response_schema=Plan,
        ),
    )
    p = Plan.model_validate_json(r.text)  # max steps and tool names are checked here, by the schema
    return p.model_copy(update={"steps": [s.model_copy(update={"step": i}) for i, s in enumerate(p.steps, 1)]})


WRITING = re.compile(r"\b(write|summari[sz]e|compose|draft|final answer|report to the user)\b(?!-)",  # not "write-offs"
                     re.IGNORECASE)


def check_plan(p: Plan) -> list[str]:
    """What the schema can't say. (Count and tool names were already checked by Plan itself.)"""
    return [f"step {s.step} is a writing step ({s.goal!r}): that's the synthesizer's job"
            for s in p.steps if WRITING.search(s.goal)]


def show(question: str) -> Plan:
    p = plan(question)
    print(f"   Q: {question}\n   approach: {p.approach}")
    for s in p.steps:
        print(f"     {s.step}. {s.goal}\n        tools: {', '.join(s.suggested_tools)}")
    problems = check_plan(p)
    print(f"   checks: {'; '.join(problems) if problems else 'ok'}")
    return p


def main():
    header("Experiment 1: the padlock question")
    show("Why are we losing money on padlocks in BLR?")

    header("Experiment 2: the other two planted stories")
    show("Which customers are hurting our cash flow, and how much do they owe?")
    print()
    show("Why do some products keep running low?")

    header(f"Experiment 3: a huge question vs the {MAX_STEPS}-step cap")
    show("Audit everything: stock, every product's movements, all invoices and payments, all staff notes, "
         "in all three warehouses, and tell me every problem.")

    header("Experiment 4: a simple question")
    show("How many padlocks are in Bangalore?")


if __name__ == "__main__":
    main()
