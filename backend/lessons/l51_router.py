"""
Lesson 5.1: Intent classifier
==============================

Run from backend/:   uv run python -m lessons.l51_router

Not every message needs the agent loop. A ROUTER is one cheap model call, with no tools,
that labels the message so CODE can pick the path (5.2 does the dispatch):

    simple_read    one fact or one list, 1-2 tool calls      -> agent loop (M1)
    investigation  "why", compare, explain, several steps    -> orchestrator (M7)
    clarify        too vague to act on, even with tools      -> ask the user (M3)
    smalltalk      greetings, thanks, "what can you do?"     -> reply, no tools, no data

The output is structured: {reason, route, confidence}.
    reason first   the model writes its reasoning BEFORE it picks; the label follows from it
    confidence     0-1, the model's own guess. 5.2 uses it: low confidence -> a safer path

How this differs from 4.5's router: 4.5 picked WHICH TOOLS to send. This one picks
WHICH PATH runs at all. A "hi" should never reach the agent loop.

Experiments:
  1. Free text: ask for the route without a schema. What does code get back?
  2. Structured: 10 hand-written questions, labelled
"""

import time
from typing import Literal

from pydantic import BaseModel, Field

from app.llm import MODEL, client, types
from lessons.common import header

ROUTER_PROMPT_V1 = (
    "You route messages sent to a warehouse assistant. The assistant can look up stock, "
    "low stock, stock movements, invoices, payments and staff notes for warehouses BLR, MUM and DEL. "
    "Decide which path should handle the message. Do not answer it."
)
ROUTER_PROMPT = ROUTER_PROMPT_V1 + (  # v2: rules added in 5.3 from the eval's mistakes (92% -> 100%)
    "\n\nRules:"
    "\n- Advice or a decision ('should I...', 'do we need to...') is an investigation: it needs evidence, then judgement."
    "\n- A question across MANY records (all customers, every product, rankings, totals with conditions) "
    "is an investigation, even if it sounds short."
    "\n- 'How is it going' / 'what's the status' with no subject is clarify, not smalltalk: the user wants data "
    "but hasn't said which."
    "\n- Users often write Hinglish (Hindi in Latin letters): judge the meaning, not the language."
)


class Route(BaseModel):
    reason: str = Field(description="One short sentence: what the user wants, and why that path.")
    route: Literal["simple_read", "investigation", "clarify", "smalltalk"] = Field(description=(
        "simple_read = one fact or one list, answerable with 1-2 lookups (a stock number, an invoice, low stock). "
        "investigation = needs several lookups and reasoning: why something happened, comparisons, trends, "
        "explanations. "
        "clarify = too vague to act on even with tools: no clear subject or goal ('is it ok?', 'check the numbers'). "
        "A product name that might match several products is NOT clarify: tools can search for it. "
        "smalltalk = greetings, thanks, questions about the assistant itself; needs no data."
    ))
    confidence: float = Field(ge=0, le=1, description="0-1. Below 0.6 if two routes fit about equally.")


def classify(message: str, tracer=None, model: str = MODEL, schema: type[Route] = Route,
             prompt: str = ROUTER_PROMPT) -> Route:
    """model / schema: 5.2 routes with a cheaper model and a schema that adds 4.5's fields.
    prompt: 5.3's evals compare prompt versions."""
    t0 = time.perf_counter()
    r = client.models.generate_content(
        model=model, contents=message,
        config=types.GenerateContentConfig(
            system_instruction=prompt, temperature=0,  # same message -> same route
            response_mime_type="application/json", response_schema=schema,
        ),
    )
    if tracer:  # step 0, like 4.5's router
        tracer.model_call(0, r, time.perf_counter() - t0)
    return schema.model_validate_json(r.text)


QUESTIONS = [  # (message, the route a person would pick)
    ("How many padlocks are in Bangalore?", "simple_read"),
    ("Show me invoice INV-00558", "simple_read"),
    ("What is running low in Delhi?", "simple_read"),
    ("Why is padlock stock in BLR dropping?", "investigation"),
    ("Compare padlock movements across all warehouses this quarter and explain the differences.", "investigation"),
    ("Which customers pay late, and how much do they owe us?", "investigation"),
    ("Is it fine?", "clarify"),
    ("Can you check the numbers?", "clarify"),
    ("hi, good morning!", "smalltalk"),
    ("thanks, that's all for today", "smalltalk"),
]


def main():
    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: free text, no schema")
    msg = "Why is padlock stock in BLR dropping?"
    r = client.models.generate_content(
        model=MODEL, contents=msg,
        config=types.GenerateContentConfig(
            system_instruction=ROUTER_PROMPT + " Routes: simple_read, investigation, clarify, smalltalk.",
            temperature=0,
        ),
    )
    print(f"   message: {msg!r}\n   model said: {r.text!r}")
    print(f"   code checks  text == 'investigation'  ->  {r.text.strip() == 'investigation'}")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: structured, 10 questions")
    agree = 0
    for message, expected in QUESTIONS:
        got = classify(message)
        ok = got.route == expected
        agree += ok
        print(f"\n   {'ok' if ok else 'XX'}  {got.route:<13} {got.confidence:.2f}  {message!r}")
        print(f"       reason: {got.reason}")
        if not ok:
            print(f"       expected: {expected}")
    print(f"\n   {agree}/{len(QUESTIONS)} match the expected route")


if __name__ == "__main__":
    main()
