"""
Lesson 4.7: Context isolation
==============================

Run from backend/:   uv run python -m lessons.l47_isolation

Context = what goes into one model call. Isolation = it holds ONLY this user's,
this session's data. Three walls, all in code:

  1. per USER     every tool result is filtered to the user's warehouses (2.1's rule,
                  now around ALL tools, incl. 4.6's notes search); long-term memory
                  is loaded by user_id
  2. per SESSION  a chat belongs to the user who started it; nobody else can read or
                  continue it (before: any session id worked for anyone)
  3. per SUB-TASK the router (4.5) and summarizer (4.3) get only what their job needs:
                  the question, or this session's old messages. Never tools, never other chats.

Experiments:
  1. BEFORE (4.6, no scope): Neha (Delhi only) asks what is running low
  2. AFTER: Neha and Ravi chat IN PARALLEL. Does anything of one reach the other?
  3. Ravi tries to read and continue Neha's chat
"""

from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import delete, text

from app.db import SessionLocal, engine
from app.models import ChatSession, UserMemory
from lessons.common import header
from lessons.l21_permissions import UserContext, denied, load_user
from lessons.l41_history import load_messages, new_session
from lessons.l46_memory import chat as chat_with_memory
from lessons.l46_memory import ensure_tables as ensure_memory_tables
from lessons.l46_memory import load_facts
from lessons.l13_tools import WAREHOUSE
from lessons.l61_typed_tools import DECLS_V7, TOOLS_V7, upper

NEHA, RAVI = 3, 4  # Neha: DEL only. Ravi: BLR only. No warehouse in common.


def ensure_tables():
    ensure_memory_tables()
    with engine.begin() as conn:  # chat_sessions existed before 4.7: add the owner column
        conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS user_id int REFERENCES users(id)"))


# --- Wall 1: every tool result, filtered to the user's warehouses ---------------------------

def strip(ctx: UserContext, result):
    """Drop anything from a warehouse the user can't see, wherever it appears in a result."""
    if not isinstance(result, dict):
        return result
    wh = result.get("warehouse")
    if isinstance(wh, str) and wh not in ctx.warehouse_codes:
        return {"error": "not_found"}  # like 2.1's invoices: don't even reveal it exists
    out = {}
    for k, v in result.items():
        if isinstance(v, list):
            v = [r for r in v if not (isinstance(r, dict) and r.get("warehouse") not in (None, *ctx.warehouse_codes))]
        out[k] = v
    if "count" in out:  # e.g. list_low_stock: the count must match what's left
        out["count"] = next((len(v) for v in out.values() if isinstance(v, list)), out["count"])
    return out


def scope_for(ctx: UserContext):
    """-> scope(fn): wraps one tool. Checks the warehouse ASKED for, then filters what came BACK."""
    def scope(fn):
        def call(**args):
            wh = upper(args.get("warehouse_code"))
            if wh in WAREHOUSE["enum"] and wh not in ctx.warehouse_codes:  # a real warehouse they can't see
                return denied(ctx, wh)  # anything invalid ("Bangalore") goes on: 6.1's typed check explains it
            return strip(ctx, fn(**args))
        return call
    return scope


# --- Wall 2: sessions belong to a user ---------------------------------------------------

class NotYourSession(Exception):
    pass


def owner_of(session_id: int) -> int | None:
    with SessionLocal() as db:
        s = db.get(ChatSession, session_id)
        return s.user_id if s else None


def check_owner(user_id: int, session_id: int):
    if owner_of(session_id) != user_id:
        raise NotYourSession(f"session {session_id} does not belong to user {user_id}")


def messages_for(user_id: int, session_id: int) -> list[dict]:
    check_owner(user_id, session_id)
    return load_messages(session_id)


# --- One chat turn, isolated ---------------------------------------------------------------

def who(ctx: UserContext) -> str:
    return (f"\n\nYou are talking to {ctx.name} ({ctx.role}), who can only see warehouses "
            f"{', '.join(sorted(ctx.warehouse_codes))}. Tools only return data from those.")


def chat(user_id: int, session_id: int, text: str, tracer=None, log: list | None = None, **extra) -> str:
    """**extra (4.8's on_context, 5.2's selected / extra_prompt) goes through 4.6 and 4.5 to 4.4."""
    check_owner(user_id, session_id)
    ctx = load_user(user_id)
    extra_prompt = who(ctx) + extra.pop("extra_prompt", "")
    return chat_with_memory(session_id, text, user_id, tracer=tracer, log=log,
                            extra_prompt=extra_prompt, scope=scope_for(ctx),
                            registry=TOOLS_V7, decls=DECLS_V7, **extra)  # 6.1: typed tools


# --- The test --------------------------------------------------------------------------------

def warehouses_in(result) -> set[str]:
    """Every warehouse code that appears anywhere in a tool result."""
    found = set()
    if isinstance(result, dict):
        for k, v in result.items():
            if k in ("warehouse", "warehouse_code") and isinstance(v, str):
                found.add(v)
            found |= warehouses_in(v)
    elif isinstance(result, list):
        for v in result:
            found |= warehouses_in(v)
    return found


SCRIPT = {  # each user: a private preference, then two questions without a warehouse
    NEHA: ["Remember: I call the Delhi warehouse 'Lotus'.", "What is running low?",
           "Why are padlocks going missing?"],
    RAVI: ["Remember: I want every answer to start with 'Namaste'.", "What is running low?",
           "Why are padlocks going missing?"],
}


def run_user(user_id: int) -> dict:
    sid, log, replies = new_session(user_id), [], []
    for q in SCRIPT[user_id]:
        replies.append(chat(user_id, sid, q, log=log))
    return {"session": sid, "log": log, "replies": replies}


def main():
    ensure_tables()
    with SessionLocal() as db:  # clean memories, so the run is repeatable
        db.execute(delete(UserMemory).where(UserMemory.user_id.in_([NEHA, RAVI])))
        db.commit()
    neha, ravi = load_user(NEHA), load_user(RAVI)

    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: BEFORE isolation (4.6's chat) - Neha, Delhi only, asks what is running low")
    log: list = []
    chat_with_memory(new_session(), "What is running low?", NEHA, log=log)
    seen = set().union(*(warehouses_in(r) for _, _, r in log))
    print(f"\n   warehouses in Neha's tool results: {sorted(seen)}   allowed: {sorted(neha.warehouse_codes)}")
    print(f"   -> {'LEAK: she saw other warehouses' if seen - neha.warehouse_codes else 'no leak'}")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: AFTER - Neha and Ravi chat in parallel (their output interleaves)")
    with ThreadPoolExecutor(max_workers=2) as pool:
        n_run, r_run = pool.map(run_user, [NEHA, RAVI])

    header("Experiment 2: results")
    checks = []
    for ctx, run, other in [(neha, n_run, ravi), (ravi, r_run, neha)]:
        seen = set().union(*(warehouses_in(r) for _, _, r in run["log"]))
        checks.append((f"{ctx.name}: tool results only from {sorted(ctx.warehouse_codes)}",
                       seen <= ctx.warehouse_codes, f"saw {sorted(seen)}"))
        theirs = " ".join(load_facts(other.user_id)).lower()
        mine = load_facts(ctx.user_id)
        checks.append((f"{ctx.name}: long-term memory has only their own facts",
                       bool(mine) and not any(f.lower() in theirs for f in mine), f"{mine}"))
        word = "lotus" if other.user_id == NEHA else "namaste"
        checks.append((f"{ctx.name}: replies never mention {other.name}'s '{word}'",
                       not any(word in r.lower() for r in run["replies"]), ""))
    for name, ok, detail in checks:
        print(f"   {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    print(f"\n   Neha on missing padlocks (BLR notes must be invisible to her):\n   {n_run['replies'][2][:300]}")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: Ravi tries Neha's session")
    for attempt, fn in [("read her messages", lambda: messages_for(RAVI, n_run["session"])),
                        ("continue her chat", lambda: chat(RAVI, n_run["session"], "what did she ask?"))]:
        try:
            fn()
            print(f"   {attempt}: ALLOWED (leak!)")
        except NotYourSession as e:
            print(f"   {attempt}: refused ({e})")


if __name__ == "__main__":
    main()
