"""
Lesson 4.3: Context compaction (long chats)
============================================

Run from backend/:   uv run python -m lessons.l43_compaction            (40 turns, a few minutes)
                     uv run python -m lessons.l43_compaction --turns 12  (quicker)

4.1 resends the WHOLE chat every turn. Turn 40 carries turns 1-39: slower, pricier,
and one day too big for the model.

Compaction, before each turn:
    if the not-yet-summarised messages > BUDGET tokens:
        fold the older ones into a running summary (one extra model call)
        keep the last KEEP_RECENT messages word for word
    send:  system prompt + summary  +  recent messages  +  the new question

Selections (4.2) live in chat_state, NOT in the history, so compaction can't lose them.

Experiments:
  1. A 40-turn chat: prompt tokens per call, with vs without compaction
  2. "movements for the 2nd one" at the end -> still item 2 of the list from turn 1
  3. What got lost: the summary, and a question about an exact early detail
"""

import sys

from app.db import SessionLocal, engine
from app.llm import MODEL, client, types
from app.models import Base, ChatSummary
from lessons.common import header
from lessons.l41_history import load_messages, new_session, to_history
from lessons.l42_selections import chat as chat_with_state
from lessons.l42_selections import ensure_tables as ensure_state_tables
from lessons.l42_selections import load_state, used

BUDGET = 800      # tokens of raw history allowed before compacting (small, so the demo triggers it)
KEEP_RECENT = 6   # messages always kept word for word = the last 3 turns


def ensure_tables():
    ensure_state_tables()
    Base.metadata.create_all(engine, tables=[ChatSummary.__table__])


def load_summary(session_id: int) -> dict:
    with SessionLocal() as db:
        s = db.get(ChatSummary, session_id)
        return {"summary": s.summary, "upto_id": s.upto_id} if s else {"summary": "", "upto_id": 0}


def save_summary(session_id: int, summary: dict):
    with SessionLocal() as db:
        db.merge(ChatSummary(session_id=session_id, **summary))
        db.commit()


def count_tokens(contents: list[types.Content]) -> int:
    return client.models.count_tokens(model=MODEL, contents=contents).total_tokens if contents else 0


# --- Compaction -------------------------------------------------------------------------

def summarize(old_summary: str, messages: list[dict]) -> str:
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    r = client.models.generate_content(
        model=MODEL,
        contents=f"Current summary:\n{old_summary or '(empty)'}\n\nNew messages:\n{transcript}",
        config=types.GenerateContentConfig(system_instruction=(
            "You keep the running summary of a chat between a user and a warehouse assistant. "
            "Merge the new messages into the summary. Keep what later questions may need: products "
            "with SKUs, warehouses, the numbers the user was given, decisions, open questions. "
            "Drop greetings and formatting. Plain bullet points, at most 150 words."
        )),
    )
    return r.text.strip()


def compacted_history(session_id: int) -> tuple[list[types.Content], str, bool]:
    """-> (recent messages to send, summary, did we compact this turn?)"""
    s = load_summary(session_id)
    recent = [m for m in load_messages(session_id) if m["id"] > s["upto_id"]]
    compacted = False
    if len(recent) > KEEP_RECENT and count_tokens(to_history(recent)) > BUDGET:
        old, recent = recent[:-KEEP_RECENT], recent[-KEEP_RECENT:]
        s = {"summary": summarize(s["summary"], old), "upto_id": old[-1]["id"]}
        save_summary(session_id, s)
        compacted = True
    return to_history(recent), s["summary"], compacted


def summary_prompt(summary: str) -> str:
    return f"\n\nSummary of the earlier conversation (older messages are not shown):\n{summary}" if summary else ""


# --- One chat turn: compact, then 4.2's turn (history + selections) -----------------------

def chat(session_id: int, text: str, tracer=None, log: list | None = None) -> str:
    history, summary, _ = compacted_history(session_id)
    return chat_with_state(session_id, text, tracer=tracer, log=log,
                           history=history, extra_prompt=summary_prompt(summary))


class PromptMeter:
    """A tiny tracer (1.4's hooks): remembers the prompt tokens of the FIRST model call of a turn."""
    def __init__(self):
        self.tokens = None

    def model_call(self, step, response, seconds):
        if step == 1:
            self.tokens = response.usage_metadata.prompt_token_count

    def tool_call(self, *args):
        pass

    def finish(self, **kwargs):
        pass


FILLER = [  # get_stock-style questions: they don't return product lists, so the selection stays
    "How many padlocks do we have in Bangalore?",
    "and in Mumbai?",
    "stock of sticky notes in Delhi?",
    "how many wall plugs in Mumbai?",
    "safety helmets in Bangalore?",
    "and mineral water in Delhi?",
    "how many garbage bags did we sell in Mumbai last month?",
    "padlocks in Delhi?",
]


def main():
    turns = int(sys.argv[sys.argv.index("--turns") + 1]) if "--turns" in sys.argv else 40
    ensure_tables()
    sid = new_session()

    # --- Experiment 1 ---------------------------------------------------------
    header(f"Experiment 1: a {turns}-turn chat (BUDGET {BUDGET} tokens, keep last {KEEP_RECENT} messages)")
    questions = ["show low stock items in Mumbai"]
    questions += [FILLER[i % len(FILLER)] for i in range(turns - 2)]
    questions += ["movements for the 2nd one"]

    rows = []
    for n, q in enumerate(questions, 1):
        _, _, compacted = compacted_history(sid)  # compact first, so we can report it
        raw = count_tokens(to_history(load_messages(sid)))  # what 4.1 would resend
        meter, log = PromptMeter(), []
        reply = chat(sid, q, tracer=meter, log=log)
        rows.append((n, q, meter.tokens, raw, compacted, log, reply))

    print(f"\n   {'turn':<5}{'question':<44}{'prompt tokens':>14}{'full history':>14}  compacted")
    for n, q, tokens, raw, compacted, *_ in rows:
        print(f"   {n:<5}{q[:42]:<44}{tokens or 0:>14}{raw:>14}  {'yes' if compacted else ''}")
    print(f"\n-> 'full history' keeps growing (what 4.1 would resend). 'prompt tokens' stays bounded:")
    print(f"   it drops each time old turns are folded into the summary.")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: the selection survived compaction")
    selection = load_state(sid)["selection"]
    *_, log, reply = rows[-1]
    print(f"   turn {turns}: 'movements for the 2nd one'")
    print(f"   saved item 2 = {selection[1]['sku']}   agent used = {used(log, 'list_movements')}")
    print(f"-> Turn 1's list was summarised away long ago. It still worked: chat_state is in the DB,")
    print(f"   added to the prompt every turn, and compaction never touches it.")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: what got lost?")
    print("   The running summary (all the model knows about the old turns):\n")
    print("   " + load_summary(sid)["summary"].replace("\n", "\n   "))
    print("\n   A question about an exact early detail, answered from the summary alone:")
    print(chat(sid, "Without using any tools: what was the exact name and current quantity of the "
                    "5th item in the very first low stock list you showed me?"))


if __name__ == "__main__":
    main()
