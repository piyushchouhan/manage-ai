"""
Lesson 6.2: Answer verification
================================

Run from backend/:   uv run python -m lessons.l62_verify

6.1 checked what goes INTO tools. This checks what comes OUT of the model: the answer.

A free-text answer can't be checked: "about 220 padlocks, mostly fine" has no handle.
So the answer becomes DATA, with a source for every number:

    {summary: "There are 222 padlocks in BLR...",
     facts: [{claim: "padlocks in BLR", value: 222, call_id: 2, field: "quantity"}]}
                                                    |           |
                     every tool result now carries a call_id    where in that result

Then CODE, not the model, checks every fact:
    1. re-run call #2 (same tool, same args) against the DB   -> fresh truth, nothing the model saw
    2. read the field "quantity" from it                       -> 222
    3. compare with the claimed value                          -> match, or CAUGHT
and one more: every number in the summary must be one of the checked facts ("unbacked" otherwise).

How the model is made to answer in this shape: a final_answer TOOL (typed with 6.1's
wrapper, so a malformed answer is sent back to fix), and tool mode ANY: the model must
call a tool on every step, so the only way to finish is final_answer.

Experiments:
  1. An honest answer: every fact verified
  2. A corrupted tool shows the model +30 padlocks: the model repeats it, verification catches it
  3. A corrupted answer (a wrong fact, a made-up number in the summary): caught
"""

import re

from pydantic import BaseModel, Field

from app.llm import types
from lessons.common import header
from lessons.l12_agent_loop import run_agent
from lessons.l13_tools import MAX_STEPS
from lessons.l23_injection import recording
from lessons.l61_typed_tools import CONFIG_V7, DECLS_V7, TOOLS_V7, typed


class Fact(BaseModel):
    claim: str = Field(description="The fact in a few words, e.g. 'padlocks on hand in BLR'.")
    value: float | str = Field(description="The value exactly as it appears in the tool result.")
    call_id: int = Field(ge=1, description="call_id of the tool result the value comes from.")
    field: str = Field(description=("Path to the value inside that result: keys and list positions joined by dots, "
                                    "e.g. 'quantity', 'summary_by_type.adjustment.total_qty', 'items.0.sku'."))


class Answer(BaseModel):
    summary: str = Field(description="The answer for the user. Dates as DD/MM/YYYY.")
    facts: list[Fact] = Field(description="One entry for EVERY number in the summary.")


class Finished(Exception):
    def __init__(self, answer: Answer):
        self.answer = answer


def finisher(schema: type[BaseModel]):
    """-> the final_answer tool for this answer shape, typed by 6.1 (a malformed answer goes back to fix)."""
    def final_answer(**args):
        raise Finished(schema.model_validate(args))  # like 3.3's ask_user: raising ends the loop
    return typed(final_answer, schema, schema)


FINAL_DECL = types.FunctionDeclaration(
    name="final_answer",
    description=(
        "Give your answer to the user. ALWAYS finish with this, never with plain text. "
        "Every number in the summary must also be in facts, copied exactly from a tool result, with that "
        "result's call_id and the field path. Do not put numbers you calculated yourself in the summary."
    ),
    parameters_json_schema=Answer.model_json_schema(),
)
CONFIG_V8 = CONFIG_V7.model_copy(update={
    "tools": [types.Tool(function_declarations=[*DECLS_V7.values(), FINAL_DECL])],
    "tool_config": types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
})


def numbered(registry: dict, log: list) -> dict:
    """2.3's recording + a call_id on every result, so the answer can cite it."""
    def wrap(fn):
        def call(**args):
            result = fn(**args)
            return {**result, "call_id": len(log)} if isinstance(result, dict) else result
        return call
    return {n: wrap(fn) for n, fn in recording(registry, log).items()}


def ask(question: str, registry: dict = TOOLS_V7, config=CONFIG_V8,
        schema: type[Answer] = Answer) -> tuple[Answer, list]:
    """-> (the structured answer, every tool call as (name, args, result)).
    config / schema: 6.3 passes a bigger answer shape (with calculated values) and its declaration."""
    log: list = []
    tools = numbered({**registry, "final_answer": finisher(schema)}, log)
    try:
        run_agent(question, config=config, registry=tools, max_steps=MAX_STEPS)
    except Finished as f:
        return f.answer, log
    raise RuntimeError("unreachable: in ANY mode the loop ends with final_answer or TooManySteps")


# --- Verification: code only, no model --------------------------------------------------------

def lookup(result, path: str):
    for key in path.split("."):
        result = result[int(key)] if isinstance(result, list) else result[key]
    return result


def same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 0.005
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def check_fact(fact: Fact, log: list, registry: dict) -> tuple[bool, str]:
    if not 1 <= fact.call_id <= len(log):
        return False, f"cites call #{fact.call_id}, which never happened"
    name, args, _ = log[fact.call_id - 1]
    fresh = registry[name](**args)  # re-run against the DB: the truth, not what the model was shown
    try:
        truth = lookup(fresh, fact.field)
    except (KeyError, IndexError, TypeError, ValueError):
        return False, f"{name} has no field {fact.field!r}"
    return (True, f"DB: {truth}") if same(truth, fact.value) else (False, f"CAUGHT: DB says {truth}")


NOT_NUMBERS = r"\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|[A-Z]{2,}-\d+"  # dates, SKUs, invoice numbers


def numbers_in(text: str) -> list[float]:
    text = re.sub(NOT_NUMBERS, " ", text)
    return [float(n.replace(",", "")) for n in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])", text)]


def unbacked(answer: Answer, also: tuple = ()) -> list[float]:
    """also: more values that count as backed (6.3's calculated values, after their maths is checked)."""
    backed = [f.value for f in answer.facts] + list(also)
    return [n for n in numbers_in(answer.summary) if not any(same(n, v) for v in backed)]


def verify(answer: Answer, log: list, registry: dict = TOOLS_V7) -> bool:
    """Print every check; True only if all facts match the DB and no number is unbacked."""
    ok = True
    for f in answer.facts:
        good, detail = check_fact(f, log, registry)
        ok &= good
        src = f"call #{f.call_id} {log[f.call_id - 1][0]}.{f.field}" if 1 <= f.call_id <= len(log) else f"call #{f.call_id}"
        print(f"     {'ok' if good else 'XX'}  {f.claim} = {f.value}   [{src}]  {detail}")
    if extra := unbacked(answer):
        ok = False
        print(f"     XX  numbers in the summary with no fact behind them: {extra}")
    print(f"   -> {'VERIFIED' if ok else 'NOT VERIFIED: do not show this answer as is'}")
    return ok


# --- Experiments --------------------------------------------------------------------------------

def inflate(fn):
    """A broken tool: shows the model 30 more than the DB has."""
    def call(**args):
        r = fn(**args)
        return {**r, "quantity": r["quantity"] + 30} if "quantity" in r else r
    return call


QUESTION = "How many padlocks are in Bangalore, and how many adjustment movements have there been?"


def main():
    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: an honest answer")
    answer, log = ask(QUESTION)
    print(f"\n   summary: {answer.summary}\n")
    verify(answer, log)

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: a corrupted tool (+30 padlocks in what the model sees)")
    bad, bad_log = ask(QUESTION, {**TOOLS_V7, "get_stock": inflate(TOOLS_V7["get_stock"])})
    print(f"\n   summary: {bad.summary}\n")
    verify(bad, bad_log)  # re-runs with the REAL tools

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: a corrupted answer (experiment 1's, edited)")
    f0 = answer.facts[0]
    wrong = f0.model_copy(update={"value": float(f0.value) + 1 if numbers_in(str(f0.value)) else "something else"})
    edited = answer.model_copy(update={"facts": [wrong, *answer.facts[1:]],
                                       "summary": answer.summary + " About 17 of them are damaged."})
    print(f"   changed fact 1 to {wrong.value}, and added 'About 17 of them are damaged.' to the summary\n")
    verify(edited, log)


if __name__ == "__main__":
    main()
