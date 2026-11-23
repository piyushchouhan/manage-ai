"""
Lesson 6.3: Business rules and the retry loop
==============================================

Run from backend/:   uv run python -m lessons.l63_rules

6.2 asked "is each number TRUE?". A business rule asks "is the ANSWER acceptable?":

    facts match the DB              6.2's check, now one rule among several
    maths is right                  the model may calculate (a total, a difference), but it must
                                    list the inputs as facts: code redoes the sum
    every number is backed          no number in the summary without a fact or a checked calculation
    only invoices the user may see  every invoice named must have come from a tool, from an allowed
                                    warehouse (4.7 filters tools; this checks the OUTPUT too)
    covers every warehouse asked    asked about Bangalore and Delhi -> the answer must say both

Each rule is a plain function: (answer, tool log, context) -> list of problems. No model.

When a rule fails:
    attempt 1 fails -> the problems go back to the model -> attempt 2 (ONE retry)
    attempt 2 fails -> SAFE FAILURE: "I couldn't verify this", plus only the facts that passed
Never show an answer that failed its checks. One retry, not a loop: if the model can't fix it
once, more tries mostly burn money.

Experiments:
  1. The rules on a real answer, then on edited copies that break each rule
  2. A tool that is wrong ONCE (+30 padlocks): attempt 1 fails, the retry passes
  3. get_stock ALWAYS wrong: can the model find another way (movements), with its maths checked?
  4. get_stock always wrong AND no other way: both attempts fail -> safe failure
"""

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from app.llm import types
from lessons.common import header
from lessons.l12_agent_loop import TooManySteps
from lessons.l21_permissions import UserContext, load_user
from lessons.l47_isolation import RAVI, scope_for, who
from lessons.l61_typed_tools import DECLS_V7, TOOLS_V7
from lessons.l62_verify import (CONFIG_V8, FINAL_DECL, Answer, BaseModel, ask, check_fact, inflate, same,
                                unbacked)

ASHA = 1


# --- The answer shape: 6.2's + calculated values --------------------------------------------

class Computed(BaseModel):
    claim: str = Field(description="What was calculated, e.g. 'padlocks in BLR and DEL together'.")
    value: float
    op: Literal["sum", "difference"] = Field(description="difference = the first fact minus the others.")
    of: list[int] = Field(min_length=2, description="The fact numbers used (1 = the first fact in facts).")


class AnswerV9(Answer):
    computed: list[Computed] = Field(default_factory=list, description=(
        "Every number in the summary you calculated yourself, with the facts it was calculated from."))


FINAL_DECL_V9 = types.FunctionDeclaration(
    name="final_answer",
    description=FINAL_DECL.description.replace(
        "Do not put numbers you calculated yourself in the summary.",
        "If you calculate a number (a total, a difference), put it in computed with the fact numbers it uses."),
    parameters_json_schema=AnswerV9.model_json_schema(),
)
CONFIG_V9 = CONFIG_V8.model_copy(update={"tools": [types.Tool(function_declarations=[*DECLS_V7.values(), FINAL_DECL_V9])]})


# --- The rules: plain code ---------------------------------------------------------------------

@dataclass
class Context:
    question: str
    user: UserContext
    registry: dict  # the CLEAN tools, scoped to the user: the truth to check against


NAMES = {"BLR": r"blr|bangalore|bengaluru", "MUM": r"mum|mumbai|bombay", "DEL": r"del|delhi"}


def warehouses_named(text: str) -> set[str]:
    return {code for code, pattern in NAMES.items() if re.search(rf"\b({pattern})\b", text, re.IGNORECASE)}


def facts_true(a: AnswerV9, log: list, c: Context) -> list[str]:
    out = []
    for i, f in enumerate(a.facts, 1):
        good, detail = check_fact(f, log, c.registry)
        if not good:
            out.append(f"fact {i} ({f.claim} = {f.value}): {detail}")
    return out


def maths_right(a: AnswerV9, log: list, c: Context) -> list[str]:
    out = []
    for comp in a.computed:
        if any(not 1 <= i <= len(a.facts) for i in comp.of):
            out.append(f"'{comp.claim}' uses fact numbers {comp.of}, but there are only {len(a.facts)} facts")
            continue
        try:
            vals = [float(a.facts[i - 1].value) for i in comp.of]
        except ValueError:
            out.append(f"'{comp.claim}' is calculated from facts that are not numbers")
            continue
        expected = sum(vals) if comp.op == "sum" else vals[0] - sum(vals[1:])
        if not same(expected, comp.value):
            out.append(f"'{comp.claim}' = {comp.value}, but the {comp.op} of facts {comp.of} is {expected:g}")
    return out


def numbers_backed(a: AnswerV9, log: list, c: Context) -> list[str]:
    extra = unbacked(a, also=tuple(comp.value for comp in a.computed))
    return [f"the summary states {n:g}, which is not a fact or a calculation" for n in extra]


def only_visible(a: AnswerV9, log: list, c: Context) -> list[str]:
    looked_up = {r["number"]: r.get("warehouse") for name, _, r in log if name == "get_invoice" and "number" in r}
    out = []
    for inv in sorted(set(re.findall(r"INV-\d{5}", a.summary))):
        if inv not in looked_up:
            out.append(f"mentions {inv}, which no tool returned")
        elif looked_up[inv] not in c.user.warehouse_codes:
            out.append(f"mentions {inv} from {looked_up[inv]}, which {c.user.name} may not see")
    return out


def covers_warehouses(a: AnswerV9, log: list, c: Context) -> list[str]:
    missing = warehouses_named(c.question) - warehouses_named(a.summary)
    return [f"the question asks about {wh}, the answer doesn't say anything about it" for wh in sorted(missing)]


RULES = {
    "facts match the DB": facts_true,
    "maths is right": maths_right,
    "every number is backed": numbers_backed,
    "only invoices the user may see": only_visible,
    "covers every warehouse asked": covers_warehouses,
}


def check(a: AnswerV9, log: list, c: Context, rules: dict = RULES) -> list[str]:
    return [f"{name}: {p}" for name, rule in rules.items() for p in rule(a, log, c)]


# --- The retry loop --------------------------------------------------------------------------------

def setup(question: str, user_id: int, broken: dict | None = None, config=CONFIG_V9):
    """-> (tools the agent uses, its config, the rules' context). broken: tools replaced for a test.
    config: 9.1 passes a bigger final_answer (with blocks)."""
    user = load_user(user_id)
    scope = scope_for(user)
    clean = {n: scope(fn) for n, fn in TOOLS_V7.items()}
    tools = {**clean, **{n: scope(fn) for n, fn in (broken or {}).items()}}
    config = config.model_copy(update={"system_instruction": config.system_instruction + who(user)})
    return tools, config, Context(question, user, clean)


def retry_prompt(question: str, a: AnswerV9, problems: list[str]) -> str:
    return (f"{question}\n\nYour previous answer was REJECTED by automatic checks:\n"
            + "\n".join(f"- {p}" for p in problems)
            + f"\nRejected summary: {a.summary!r}\nLook the data up again (call ids start again) and answer again. "
              "If the data still doesn't add up, say you couldn't verify it instead of guessing.")


def safe_failure(a: AnswerV9 | None, log: list, c: Context) -> str:
    ok = [f for f in a.facts if check_fact(f, log, c.registry)[0]] if a else []
    text = "I couldn't produce an answer that passed my checks, so I won't guess."
    if ok:
        text += " What I could verify:\n" + "\n".join(f"- {f.claim}: {f.value:g}" if isinstance(f.value, float)
                                                     else f"- {f.claim}: {f.value}" for f in ok)
    return text


def answer(question: str, user_id: int, broken: dict | None = None, retries: int = 1) -> str:
    """-> what the user sees: the summary if every rule passed, else the safe failure."""
    a, log, c, ok = answer_checked(question, user_id, broken, retries)
    return a.summary if ok and a else safe_failure(a, log, c)


def answer_checked(question: str, user_id: int, broken: dict | None = None, retries: int = 1,
                   schema: type[AnswerV9] = AnswerV9, config=CONFIG_V9,
                   rules: dict = RULES) -> tuple[AnswerV9 | None, list, Context, bool]:
    """The retry loop. -> (the last answer, its tool log, the context, passed?).
    schema / config / rules: 9.1 asks for a bigger answer (with blocks) and adds a rule for it."""
    tools, config, c = setup(question, user_id, broken, config)
    prompt, best, best_log = question, None, []
    for attempt in range(1, retries + 2):
        try:
            a, log = ask(prompt, tools, config, schema)
        except TooManySteps:  # no answer at all is a failed attempt too, not a crash
            print(f"\n   attempt {attempt}: ran out of steps, no answer")
            prompt = f"{question}\n\nYour previous try ran out of steps. Answer with what you can verify, or say you can't."
            continue
        best, best_log = a, log
        problems = check(a, log, c, rules)
        print(f"\n   attempt {attempt}: {a.summary}")
        for p in problems:
            print(f"     XX {p}")
        if not problems:
            print("     all rules pass")
            return a, log, c, True
        prompt = retry_prompt(question, a, problems)
    print("   -> out of retries: safe failure")
    return best, best_log, c, False


# --- Experiments --------------------------------------------------------------------------------------

def first_call_only(bad, good):
    """A flaky tool: the first call goes to `bad`, every later call to `good`."""
    calls = {"n": 0}

    def call(**args):
        calls["n"] += 1
        return (bad if calls["n"] == 1 else good)(**args)
    return call


def show(label: str, a: AnswerV9, log: list, c: Context):
    problems = check(a, log, c)
    print(f"\n   {label}: {'PASS' if not problems else 'FAIL'}")
    for p in problems:
        print(f"     XX {p}")


def main():
    stock = TOOLS_V7["get_stock"]

    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: the rules on a real answer, and on broken copies of it")
    q = "How many padlocks are in Bangalore and Delhi together?"
    tools, config, c = setup(q, ASHA)
    a, log = ask(q, tools, config, AnswerV9)
    print(f"\n   summary:  {a.summary}")
    print(f"   facts:    {[(f.claim, f.value) for f in a.facts]}")
    print(f"   computed: {[(x.claim, x.value, x.op, x.of) for x in a.computed]}")
    show("the real answer", a, log, c)
    if a.computed:
        wrong = a.computed[0].model_copy(update={"value": a.computed[0].value + 10})
        show("total off by 10", a.model_copy(update={"computed": [wrong]}), log, c)
    first = a.facts[0]
    show("only one warehouse in the summary",
         a.model_copy(update={"summary": f"There are {first.value:g} padlocks.", "computed": []}), log, c)
    show("an invoice nobody looked up", a.model_copy(update={"summary": a.summary + " See INV-00558."}), log, c)
    ravi_log = log + [("get_invoice", {"number": "INV-00558"}, TOOLS_V7["get_invoice"](number="INV-00558"))]
    _, _, ravi = setup(q, RAVI)  # as if 4.7's filter had a bug and let a MUM invoice through
    leak = only_visible(a.model_copy(update={"summary": a.summary + " See INV-00558."}), ravi_log, ravi)
    print(f"\n   Ravi (BLR) shown a Mumbai invoice: {'PASS' if not leak else 'FAIL'}")
    for p in leak:
        print(f"     XX only invoices the user may see: {p}")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: get_stock is wrong ONCE (+30)")
    reply = answer("How many padlocks are in Bangalore?", ASHA,
                   broken={"get_stock": first_call_only(inflate(stock), stock)})
    print(f"\n   USER SEES: {reply}")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: get_stock is ALWAYS wrong (+30), movements still work")
    reply = answer("How many padlocks are in Bangalore and how many adjustments were there?", ASHA,
                   broken={"get_stock": inflate(stock)})
    print(f"\n   USER SEES: {reply}")

    # --- Experiment 4 ---------------------------------------------------------
    header("Experiment 4: get_stock ALWAYS wrong, and no other way to the number")
    reply = answer("How many padlocks are in Bangalore?", ASHA,
                   broken={"get_stock": inflate(stock),
                           "list_movements": lambda **args: {"error": "unavailable", "message": "movements are offline"}})
    print(f"\n   USER SEES: {reply}")


if __name__ == "__main__":
    main()
