"""
Lesson 5.2: Dispatch
=====================

Run from backend/:   uv run python -m lessons.l52_dispatch

5.1 labels a message. Dispatch is the CODE that acts on the label:

    HANDLERS = {route: function}

    smalltalk      cheap model, no tools, no data                1 cheap call
    clarify        cheap model writes ONE question back          1 cheap call
    simple_read    4.7's isolated agent, only the tools it needs router + 1-3 calls
    investigation  a separate handler: every tool, full history, an "investigate" prompt.
                   For now it is still the agent loop; M7 replaces it with planner -> workers.

Three decisions:
  1. CHEAP ROUTER   the router runs on EVERY message, so it uses ROUTER_MODEL (flash-lite).
                    One call also fills 4.5's fields (tool groups, follow_up): no second router.
  2. LAST TURN      the router sees the previous turn, so "and in Mumbai?" isn't mistaken for "too vague".
  3. FALLBACK       confidence < THRESHOLD -> investigation: the path that can do everything.
                    Being wrong there costs money; being wrong the other way costs a wrong answer.

Code rules before the router: a pending choice ("which box?") skips routing, as in 4.5.

Experiments:
  1. One chat through the dispatcher: route, path, and model calls per message (from the trace)
  2. The fallback, forced with a low-confidence decision
"""

import json
import time

from app.llm import ROUTER_MODEL, client, types
from lessons.common import header
from lessons.l14_tracing import Tracer
from lessons.l21_permissions import load_user
from lessons.l41_history import new_session, save_turn
from lessons.l42_selections import load_state
from lessons.l45_selection import GROUPS
from lessons.l45_selection import Route as ToolRoute
from lessons.l47_isolation import chat as agent_chat
from lessons.l47_isolation import check_owner, ensure_tables, messages_for, who
from lessons.l51_router import Route as PathRoute
from lessons.l51_router import classify

THRESHOLD = 0.6
ASHA = 1  # admin: sees every warehouse, so the demo isn't about permissions


class Decision(ToolRoute, PathRoute):
    """5.1's fields (reason, route, confidence) + 4.5's (groups, follow_up), from ONE call."""


# --- The router -------------------------------------------------------------------------

def format_input(last: list[dict], text: str) -> str:
    """The router's input: the last turn, cut short, and the new message. 5.3's evals use it too."""
    if not last:
        return text
    earlier = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in last)
    return f"Earlier turn (context only):\n{earlier}\n\nNew message to route: {text}"


def router_input(user_id: int, session_id: int, text: str) -> str:
    """Only what the router needs (4.7's sub-task wall): this session's last turn and the new message."""
    return format_input(messages_for(user_id, session_id)[-2:], text)


def decide(user_id: int, session_id: int, text: str, tracer=None) -> Decision:
    return classify(router_input(user_id, session_id, text), tracer, model=ROUTER_MODEL, schema=Decision)


def pick(d: Decision) -> str:
    if d.confidence < THRESHOLD:
        print(f"   [dispatch] confidence {d.confidence:.2f} < {THRESHOLD}: {d.route} -> investigation (fallback)")
        return "investigation"
    return d.route


# --- Handlers -----------------------------------------------------------------------------

CAN_DO = "You can look up stock, low stock, stock movements, invoices, payments and staff notes."


def reply_without_data(user_id: int, session_id: int, text: str, instruction: str, tracer=None) -> str:
    """smalltalk and clarify: one cheap call, no tools, so no data can be fetched or invented."""
    t0 = time.perf_counter()
    r = client.models.generate_content(
        model=ROUTER_MODEL, contents=text,
        config=types.GenerateContentConfig(
            system_instruction=f"You are a warehouse assistant. {CAN_DO}{who(load_user(user_id))}\n\n{instruction}",
        ),
    )
    if tracer:
        tracer.model_call(1, r, time.perf_counter() - t0)
    reply = r.text.strip()
    save_turn(session_id, text, reply)  # still part of the chat: the next message may answer it
    return reply


def smalltalk(user_id, session_id, text, d, tracer=None, log=None) -> str:
    return reply_without_data(user_id, session_id, text,
                              "Reply briefly and warmly. Never state numbers or facts about the data.", tracer)


def clarify(user_id, session_id, text, d, tracer=None, log=None) -> str:
    return reply_without_data(user_id, session_id, text,
                              f"The message is too vague to act on ({d.reason}). Ask ONE short question to find out "
                              "what they need, with 2-3 examples of what you can check.", tracer)


def simple_read(user_id, session_id, text, d, tracer=None, log=None) -> str:
    return agent_chat(user_id, session_id, text, tracer=tracer, log=log, selected=d)


INVESTIGATE = ("\n\nThis is an INVESTIGATION. Gather evidence from several tools (movements, notes, invoices) "
               "before you answer. Explain the cause step by step and say which tool each number came from.")


def investigation(user_id, session_id, text, d, tracer=None, log=None) -> str:
    everything = d.model_copy(update={"groups": list(GROUPS), "follow_up": True})
    return agent_chat(user_id, session_id, text, tracer=tracer, log=log, selected=everything,
                      extra_prompt=INVESTIGATE)


HANDLERS = {"smalltalk": smalltalk, "clarify": clarify, "simple_read": simple_read, "investigation": investigation}


# --- One chat turn ---------------------------------------------------------------------------

def chat(user_id: int, session_id: int, text: str, tracer=None, log: list | None = None,
         handlers: dict = HANDLERS) -> tuple[str, str]:
    """-> (reply, the path that ran). handlers: 7.3 swaps in the planner/workers investigation."""
    check_owner(user_id, session_id)
    if load_state(session_id)["pending"]:  # answering our question: no routing, the agent has the options
        print("   [dispatch] pending choice -> simple_read, no router")
        return agent_chat(user_id, session_id, text, tracer=tracer, log=log), "simple_read (pending)"
    d = decide(user_id, session_id, text, tracer)
    path = pick(d)
    print(f"   [dispatch] {d.route} ({d.confidence:.2f}) -> {path}: {d.reason}")
    return handlers[path](user_id, session_id, text, d, tracer=tracer, log=log), path


# --- Experiments ------------------------------------------------------------------------------

def calls(tracer: Tracer) -> tuple[int, int]:
    """-> (cheap-model calls, main-model calls), read from 1.4's trace file."""
    models = [e["model"] for e in map(json.loads, tracer.path.read_text().splitlines()) if e["event"] == "model_call"]
    cheap = sum(1 for m in models if m and ROUTER_MODEL in m)
    return cheap, len(models) - cheap


CHAT = [
    "hi!",
    "Can you check the numbers?",
    "What is running low in Delhi?",
    "Show me invoice INV-00558",
    "How many padlocks are in Bangalore?",
    "Why is padlock stock in BLR dropping?",
    "and in Mumbai?",
    "thanks, that's all",
]


def main():
    ensure_tables()

    # --- Experiment 1 ---------------------------------------------------------
    header(f"Experiment 1: one chat through the dispatcher (router: {ROUTER_MODEL})")
    sid, rows = new_session(ASHA), []
    for text in CHAT:
        tracer = Tracer(text)
        reply, path = chat(ASHA, sid, text, tracer=tracer)
        print(f"A: {reply[:200]}")
        rows.append((text, path, *calls(tracer)))

    header("Experiment 1: results")
    print(f"   {'message':<40} {'path':<22} cheap  main")
    for text, path, cheap, main_calls in rows:
        print(f"   {text[:40]:<40} {path:<22} {cheap:>5}  {main_calls:>4}")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: the fallback")
    unsure = Decision(reason="could be one lookup or a comparison", route="simple_read", confidence=0.45,
                      groups=["stock"], follow_up=False)
    print(f"   pick(route=simple_read, confidence=0.45) -> {pick(unsure)}")
    sure = unsure.model_copy(update={"confidence": 0.9})
    print(f"   pick(route=simple_read, confidence=0.90) -> {pick(sure)}")


if __name__ == "__main__":
    main()
