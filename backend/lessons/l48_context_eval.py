"""
Lesson 4.8: Context evaluation
===============================

Run from backend/:   uv run python -m lessons.l48_context_eval

Until now we judged the ANSWER. But the answer is only as good as what the model was
shown. This lesson tests the CONTEXT itself: the exact system prompt + messages + tool
results of every model call, captured by 4.4's build_context() through its on_context hook.

Three checks per test question:
  1. NEEDED     the facts the answer depends on are in the context of the LAST call
                (the one that writes the answer): a stock number, a memory, the selection...
  2. FORBIDDEN  nothing that must not be there appears in ANY call: another user's memory
                or chat, notes or rows from a warehouse the user can't see (4.7's walls)
  3. BUDGET     every call keeps memory / history / work within 4.4's budgets

Each case starts clean: no long-term memories, fresh session. Setup is done in code
(seeded facts, earlier turns, other users' chats) so the case tests one thing.
"""

import json
from dataclasses import dataclass, field

from sqlalchemy import delete

from app.db import SessionLocal, query
from app.models import UserMemory
from lessons.common import header
from lessons.l21_permissions import load_user
from lessons.l41_history import new_session
from lessons.l42_selections import list_low_stock
from lessons.l44_budget import BUDGET, part_text
from lessons.l46_memory import remember
from lessons.l47_isolation import NEHA, RAVI, chat, ensure_tables, warehouses_in

ASHA = 1  # admin: all warehouses
USERS = [ASHA, NEHA, RAVI]


@dataclass
class Case:
    name: str
    user: int
    question: str
    need: list[str]                                      # must be in the LAST call's context
    forbid: list[str] = field(default_factory=list)      # must be in NO call's context
    facts: dict[int, list[str]] = field(default_factory=dict)  # long-term memories to seed, per user
    earlier: list[str] = field(default_factory=list)     # earlier turns, same user, same session
    others: list[tuple[int, str]] = field(default_factory=list)  # other users' chats, run first


def padlocks(wh: str) -> int:
    return query("""SELECT s.quantity FROM stock_levels s JOIN products p ON p.id = s.product_id
                    JOIN warehouses w ON w.id = s.warehouse_id
                    WHERE p.sku = 'HAR-0018' AND w.code = :wh""", wh=wh)[0]["quantity"]


def cases() -> list[Case]:
    """Expected values come from the DB, not typed by hand: the test stays true if the data changes."""
    blr = padlocks("BLR")
    low = [r["sku"] for r in list_low_stock("BLR")["items"]]
    blr_notes = ["5 padlocks short", "no camera coverage"]  # ops_notes #11, #13 (BLR)
    return [
        Case("tool result reaches the model", RAVI, "How many padlocks are in Bangalore?",
             need=[f'"quantity": {blr}']),
        Case("denied, and no BLR number leaks", NEHA, "How many padlocks are in Bangalore?",
             need=["access_denied"], forbid=[f'"quantity": {blr}']),
        Case("long-term memory is loaded", NEHA, "How many padlocks are in Lotus?",
             facts={NEHA: ["Calls the Delhi warehouse 'Lotus'."]}, need=["Lotus"]),
        Case("another user's memory stays out", RAVI, "What is running low?",
             facts={NEHA: ["Calls the Delhi warehouse 'Lotus'."]}, need=low, forbid=["Lotus"]),
        Case("history carries the product", RAVI, "Why is it dropping?",
             earlier=["How many padlocks are in Bangalore?"], need=["HAR-0018"]),
        Case("selection resolves 'the 2nd one'", RAVI, "Show recent movements for the 2nd one.",
             earlier=["What is running low?"], need=[f"2. {low[1]}"]),
        Case("pending clarification is in context", RAVI, "the first one",
             earlier=["stock of box"], need=["You asked the user", "PAC-0001"]),
        Case("notes filtered to own warehouse", NEHA, "Why are padlocks going missing?",
             need=["can only see warehouses DEL"], forbid=blr_notes),
        Case("another user's chat stays out", RAVI, "What supplier code was mentioned earlier?",
             others=[(NEHA, "Note for later: our new supplier code is ZEBRA-42.")],
             need=["can only see warehouses BLR"], forbid=["ZEBRA-42"]),
        Case("a huge result fits the budget", ASHA, "List the last 50 padlock movements in Bangalore, every row.",
             need=["trimmed"]),
    ]


# --- Capture -----------------------------------------------------------------------------

def capture(calls: list):
    """-> on_context hook: saves what ONE model call was shown."""
    def on_context(contents, config, parts):
        results = [p.function_response.response for c in contents for p in (c.parts or []) if p.function_response]
        calls.append({
            "text": config.system_instruction + "\n" + "\n".join(part_text(p) for c in contents for p in (c.parts or [])),
            "warehouses": set().union(*(warehouses_in(r) for r in results)),
            "parts": parts,
        })
    return on_context


def reset():
    with SessionLocal() as db:
        db.execute(delete(UserMemory).where(UserMemory.user_id.in_(USERS)))
        db.commit()


def run(case: Case) -> list[dict]:
    reset()
    for uid, facts in case.facts.items():
        for f in facts:
            remember(uid, f)
    for uid, text in case.others:
        chat(uid, new_session(uid), text)
    sid = new_session(case.user)
    for text in case.earlier:
        chat(case.user, sid, text)
    calls: list = []  # only the question under test is captured
    chat(case.user, sid, case.question, on_context=capture(calls))
    return calls


# --- Checks ------------------------------------------------------------------------------

def check(case: Case, calls: list[dict]) -> dict[str, tuple[bool, str]]:
    if not calls:
        return {"needed": (False, "no model call captured"), "forbidden": (False, ""), "budget": (False, "")}
    allowed = load_user(case.user).warehouse_codes
    missing = [n for n in case.need if n not in calls[-1]["text"]]
    leaked = [f for f in case.forbid if any(f in c["text"] for c in calls)]
    leaked += [f"rows from {w}" for w in sorted(set().union(*(c["warehouses"] for c in calls)) - allowed)]
    peak = {k: max(c["parts"][k] for c in calls) for k in BUDGET}
    over = [f"{k} {peak[k]}>{BUDGET[k]}" for k in BUDGET if peak[k] > BUDGET[k]]
    return {
        "needed": (not missing, f"missing {missing}" if missing else ""),
        "forbidden": (not leaked, f"leaked {leaked}" if leaked else ""),
        "budget": (not over, ", ".join(over) or "peak " + json.dumps(peak)),
    }


def main():
    ensure_tables()
    results = []
    for i, case in enumerate(cases(), 1):
        header(f"Case {i}: {case.name}  ({load_user(case.user).name}: {case.question!r})")
        calls = run(case)
        results.append((case, len(calls), check(case, calls)))
    reset()

    header("Context scorecard")
    passed = 0
    for i, (case, n, checks) in enumerate(results, 1):
        ok = all(good for good, _ in checks.values())
        passed += ok
        print(f"\n   {i:>2}. {'PASS' if ok else 'FAIL'}  {case.name}  ({n} model calls)")
        for name, (good, detail) in checks.items():
            print(f"         {'ok ' if good else 'XX '} {name:<9} {detail}")
    print(f"\n   {passed}/{len(results)} cases pass all three checks")


if __name__ == "__main__":
    main()
