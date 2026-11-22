"""
Lesson 4.5: Context selection
==============================

Run from backend/:   uv run python -m lessons.l45_selection

4.4 showed where the tokens go: the 7 tool declarations are ~1,500 of ~1,600 tokens,
and they are resent on EVERY step of EVERY question, needed or not.

Selection = send only what this question needs. A small, cheap ROUTER call first
(structured output, no tools) decides:
    groups:    which tool groups the question needs      -> only those declarations
    follow_up: does it only make sense with the chat?    -> else only the last turn, no summary
Code then builds the context from that. Safety rule in code: if a choice is pending
("which box?"), skip the router and send everything ("2" can't be routed).

Rows were already selected in 2.2 (compact results, limit) and trimmed in 4.4.

Experiments:
  1. The same 6-question chat, full context vs selected: same tool calls? tokens from traces?
  2. Selecting too little: "and in Mumbai?" with no history at all
"""

import json
import time
from typing import Literal

from pydantic import BaseModel, Field

from app.llm import MODEL, client, types
from lessons.common import header
from lessons.l14_tracing import Tracer
from lessons.l41_history import new_session
from lessons.l42_selections import load_state
from lessons.l43_compaction import ensure_tables
from lessons.l44_budget import chat as chat_with_budget

ALWAYS = ["search_products", "ask_user"]  # names -> SKUs, and asking back: any question may need them
GROUPS = {
    "stock": ["get_stock", "list_low_stock", "list_suppliers"],  # list_suppliers: added in 7.3
    "movements": ["list_movements", "resolve_period"],
    "invoices": ["get_invoice", "list_unpaid_invoices"],  # the 2nd one: added in 7.2 (V7 tools only)
}


class Route(BaseModel):
    groups: list[Literal["stock", "movements", "invoices"]] = Field(description=(
        "stock = current quantities, low stock, suppliers; movements = history of stock changes, sales, "
        "purchases, why stock changed, anything with a time period; invoices = who bought, "
        "payments, invoice details. Pick every group the question could need."
    ))
    follow_up: bool = Field(description=(
        "True if the message only makes sense with the earlier chat: 'and in Mumbai?', 'why?', "
        "'the 2nd one', 'those'. False if it stands on its own."
    ))


def route(question: str, tracer=None) -> Route:
    t0 = time.perf_counter()
    r = client.models.generate_content(
        model=MODEL, contents=question,
        config=types.GenerateContentConfig(
            system_instruction="Classify a message sent to a warehouse assistant.",
            response_mime_type="application/json", response_schema=Route,
        ),
    )
    if tracer:  # step 0 in the trace, so the router's tokens are counted too
        tracer.model_call(0, r, time.perf_counter() - t0)
    return Route.model_validate_json(r.text)


def chat(session_id: int, text: str, tracer=None, log: list | None = None,
         selected: Route | None = None, **extra) -> str:
    """4.4's turn, with only the tools and history this question needs.
    selected: a route already decided (5.2's router fills these fields too), so we don't route twice.
    **extra (extra_prompt, extra_tools from 4.6) is passed straight through."""
    if load_state(session_id)["pending"]:  # the user is answering our question: send everything
        print("   [route] pending choice -> all tools + history")
        return chat_with_budget(session_id, text, tracer=tracer, log=log, **extra)
    r = selected or route(text, tracer)
    tools = ALWAYS + [t for g in r.groups for t in GROUPS[g]]
    print(f"   [route] groups={r.groups} follow_up={r.follow_up} -> tools {tools}")
    return chat_with_budget(session_id, text, tracer=tracer, log=log, tools=tools, use_history=r.follow_up,
                            **extra)


# --- Measuring from the trace files (1.4) --------------------------------------------------

def from_trace(tracer: Tracer) -> tuple[int, list[str]]:
    """-> (total prompt tokens of all model calls incl. the router, the tool calls made)"""
    events = [json.loads(line) for line in tracer.path.read_text().splitlines()]
    total = sum(e["prompt_tokens"] for e in events if e["event"] == "model_call")
    calls = [f"{e['tool']}({','.join(str(v) for _, v in sorted(e['args'].items()))})"
             for e in events if e["event"] == "tool_call"]
    return total, calls


TEST_CHAT = [
    "How many padlocks do we have in Bangalore?",
    "and in Mumbai?",
    "Why is padlock stock in BLR dropping?",
    "Who bought padlocks from BLR most recently, and have they paid?",
    "What is running low in Delhi?",
    "How many padlocks did we sell in BLR last month?",
]


def run(chat_fn, label: str) -> list[tuple[int, list[str], str]]:
    header(f"Running the test chat: {label}")
    sid, rows = new_session(), []
    for q in TEST_CHAT:
        t = Tracer(q)
        reply = chat_fn(sid, q, tracer=t)
        rows.append((*from_trace(t), reply))
    return rows


def main():
    ensure_tables()

    # --- Experiment 1 ---------------------------------------------------------
    full = run(chat_with_budget, "FULL context (4.4)")
    sel = run(chat, "SELECTED context (4.5)")

    header("Experiment 1: full vs selected (prompt tokens summed from the trace files)")
    print(f"   {'question':<46}{'full':>7}{'selected':>10}   same tool calls?")
    for q, (ft, fc, _), (st, sc, _) in zip(TEST_CHAT, full, sel):
        print(f"   {q[:44]:<46}{ft:>7}{st:>10}   {'yes' if sorted(fc) == sorted(sc) else 'NO'}")
        if sorted(fc) != sorted(sc):
            print(f"      full:     {fc}\n      selected: {sc}")
    f_total, s_total = sum(r[0] for r in full), sum(r[0] for r in sel)
    print(f"\n   total: full {f_total}, selected {s_total} (router calls included)"
          f"  -> {100 * (f_total - s_total) / f_total:.0f}% fewer prompt tokens")
    print("\n   Answers side by side (first line):")
    for q, (*_, fa), (*_, sa) in zip(TEST_CHAT, full, sel):
        print(f"   Q {q}\n     full:     {fa.splitlines()[0][:100]}\n     selected: {sa.splitlines()[0][:100]}")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: selecting too little -> 'and in Mumbai?' with no history")
    print(chat_with_budget(new_session(), "and in Mumbai?"))  # a new chat = no history at all
    print("\n-> Without the earlier turn, 'and in Mumbai?' has no product. The router's follow_up")
    print("   flag exists to prevent exactly this; if the router gets it wrong, the answer breaks.")


if __name__ == "__main__":
    main()
