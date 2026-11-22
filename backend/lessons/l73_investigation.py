"""
Lesson 7.3: End-to-end investigation
=====================================

Run from backend/:   uv run python -m lessons.l73_investigation

Every piece exists. This lesson connects them behind ONE entrance, the chat:

    message
      -> 5.2 router (cheap model)          smalltalk / clarify / simple_read: as before
      -> investigation:
           7.1 planner                     1 call, no tools -> checked plan
           7.2 workers                     one small agent loop per step, shared numbered log
           7.2 synthesizer                 1 call, no tools -> report with cited facts
           VALIDATOR (code)                6.2 every fact re-run against the DB
                                           6.2 no number without a fact
                                           6.3 only invoices the user may see, every warehouse asked
             pass -> reply + sources
             fail -> ONE rewrite with the problems listed (same evidence, no new tool calls)
             fail again -> 6.3's safe failure: only what could be verified

New in 7.3: list_suppliers (6.1 typed, 4.7 scoped). Without it the third story was invisible.

Grading: the "right answer" for each planted story comes from SQL written here, NOT from the
agent's tools. A story passes when the reply names the right things, shows the key number, and
went through the validator.

Experiments:
  1. One chat: a simple read, then the three planted stories (shrinkage, non-paying customers,
     slow supplier), every message through the router
  2. The scorecard
"""

import time
from dataclasses import dataclass

from app.db import query
from lessons.common import header
from lessons.l14_tracing import Tracer
from lessons.l21_permissions import load_user
from lessons.l41_history import new_session, save_turn
from lessons.l47_isolation import ensure_tables
from lessons.l52_dispatch import HANDLERS, calls
from lessons.l52_dispatch import chat as dispatch_chat
from lessons.l62_verify import numbers_in, same
from lessons.l63_rules import RULES, Context, safe_failure
from lessons.l72_workers import Report, gather, show_report, synthesize

ASHA = 1
RETRIES = 1  # one rewrite: the synthesizer usually fixes a bad citation; a second try rarely helps
EXTRA_RULES = {k: RULES[k] for k in ("only invoices the user may see", "covers every warehouse asked")}


# --- The validator ---------------------------------------------------------------------------------

def fmt(v) -> str:
    return f"{v:,.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def with_sources(report: Report, log: list) -> str:
    lines = [f"- [step {f.step} · {log[f.call_id - 1][0]}] {f.claim}: {fmt(f.value)}" for f in report.facts]
    return report.summary + "\n\nSources:\n" + "\n".join(lines)


def investigate(question: str, user_id: int, retries: int = RETRIES) -> tuple[str, bool]:
    """-> (reply for the user, verified?)"""
    t0 = time.perf_counter()
    inv = gather(question, user_id)
    c = Context(question, load_user(user_id), inv.clean)
    feedback, report = "", None
    for attempt in range(1, retries + 2):
        report = synthesize(question, inv.evidence, inv.log, inv.who_text, feedback)
        problems = show_report(report, inv.evidence, inv.log, inv.clean)
        problems += [f"{name}: {p}" for name, rule in EXTRA_RULES.items() for p in rule(report, inv.log, c)]
        took = f"{len(inv.plan.steps)} steps · {len(inv.log)} tool calls · {time.perf_counter() - t0:.0f}s"
        if not problems:
            print(f"\n   attempt {attempt}: VERIFIED · {took}")
            return with_sources(report, inv.log), True
        print(f"\n   attempt {attempt}: REJECTED · {took}")
        for p in problems:
            print(f"     XX {p}")
        feedback = ("\n\nYour previous report was REJECTED by automatic checks:\n" + "\n".join(f"- {p}" for p in problems)
                    + "\nWrite it again from the same evidence. Fix or drop every claim you can't cite exactly.")
    print("   -> out of retries: safe failure")
    return safe_failure(report, inv.log, c), False


def investigation(user_id, session_id, text, d, tracer=None, log=None) -> str:
    """5.2's investigation handler, now planner -> workers -> synthesizer -> validator."""
    reply, _ = investigate(text, user_id)
    save_turn(session_id, text, reply)
    return reply


HANDLERS_V2 = {**HANDLERS, "investigation": investigation}


def chat(user_id: int, session_id: int, text: str, tracer=None, log: list | None = None) -> tuple[str, str]:
    return dispatch_chat(user_id, session_id, text, tracer=tracer, log=log, handlers=HANDLERS_V2)


# --- The planted stories: the truth from SQL, not from the agent's tools ----------------------------

@dataclass
class Story:
    name: str
    question: str
    mention: list[tuple[str, ...]]  # the reply must contain one of each group
    numbers: list[float]            # ...and show each of these


def stories() -> list[Story]:
    shrink = query("""SELECT p.sku, SUM(m.quantity) AS total
                      FROM stock_movements m JOIN products p ON p.id = m.product_id
                      JOIN warehouses w ON w.id = m.warehouse_id
                      WHERE w.code = 'BLR' AND m.movement_type = 'adjustment'
                      GROUP BY p.sku ORDER BY total LIMIT 1""")[0]
    never = query("""SELECT c.name, c.credit_limit FROM customers c
                     WHERE EXISTS (SELECT 1 FROM invoices i WHERE i.customer_id = c.id AND i.status <> 'cancelled')
                       AND NOT EXISTS (SELECT 1 FROM payments p JOIN invoices i ON i.id = p.invoice_id
                                       WHERE i.customer_id = c.id)""")
    slow = query("SELECT name, lead_time_days FROM suppliers ORDER BY lead_time_days DESC LIMIT 1")[0]
    return [
        Story("shrinkage", "Why are we losing money on padlocks in BLR?",
              [(shrink["sku"], "padlock"), ("adjustment",)], [shrink["total"]]),
        Story("non-paying customers", "Which customers are hurting our cash flow, and how much do they owe?",
              [(c["name"],) for c in never], sorted({c["credit_limit"] for c in never})),
        Story("slow supplier", "Why do some of our products keep running low?",
              [(slow["name"],)], [slow["lead_time_days"]]),
    ]


def grade(s: Story, reply: str, path: str) -> list[str]:
    """-> what's missing (empty = the story was found, with the right numbers)."""
    low, shown = reply.lower(), numbers_in(reply)
    out = [] if path == "investigation" else [f"routed to {path}, not investigation"]
    out += [f"doesn't mention {' / '.join(g)}" for g in s.mention if not any(x.lower() in low for x in g)]
    out += [f"doesn't show {fmt(float(n))}" for n in s.numbers if not any(same(n, v) for v in shown)]
    if reply.startswith("I couldn't produce"):
        out.append("did not pass the validator (safe failure)")
    return out


def main():
    ensure_tables()
    sid = new_session(ASHA)
    plot = stories()

    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: a simple read through the same entrance")
    t0, tracer = time.perf_counter(), Tracer("simple")
    reply, path = chat(ASHA, sid, "How many padlocks are in Bangalore?", tracer=tracer)
    cheap, main_calls = calls(tracer)
    print(f"A: {reply}\n   path={path} · {cheap} cheap + {main_calls} main model calls · {time.perf_counter() - t0:.0f}s")

    rows = []
    for s in plot:
        header(f"Experiment 1: {s.name}\n   Q: {s.question}")
        t0 = time.perf_counter()
        reply, path = chat(ASHA, sid, s.question)
        print(f"\n   REPLY ({path}, {time.perf_counter() - t0:.0f}s):\n{reply}")
        rows.append((s, grade(s, reply, path)))

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: scorecard (truth from SQL)")
    for s, missing in rows:
        want = "; ".join(" / ".join(g) for g in s.mention) + " · " + ", ".join(fmt(float(n)) for n in s.numbers)
        print(f"   {'FOUND ' if not missing else 'MISSED'}  {s.name:<22} expected: {want}")
        for m in missing:
            print(f"            XX {m}")
    print(f"\n   {sum(not m for _, m in rows)}/{len(rows)} planted stories found with correct numbers")


if __name__ == "__main__":
    main()
