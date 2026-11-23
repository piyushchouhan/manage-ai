"""
Lesson 9.2: Rendering in shadcn
================================

Run from backend/:   uv run python -m lessons.l92_rendering

9.1 built {summary, blocks, follow_ups} with 6.3's strict loop. The chat can't use that loop as it is:
the chat has history, memory, selections and streaming (M4, M8). So the chat gets blocks a lighter way,
AFTER its normal reply, from the same tool log:

    simple read:  agent reply (streamed, as before)          "Mumbai has 5 products below reorder level..."
                  + LAYOUT: one cheap call picks blocks       {kind: table, call_id: 1, path: "items", columns}
                    pointing at this turn's tool results
                  + 9.1's build(): code copies the rows       Table(rows from list_low_stock, options = SKUs)
                  -> saved with the message (chat_messages.display), sent as an SSE "display" event

    a bad pointer is DROPPED, not retried: the reply text is already out, a block is extra. The data in a
    block still can't be wrong: it is copied from the tool result, never written by the model.

The UI (frontend/src/App.tsx) renders:
    table      shadcn <Table>; rows with a sku are clickable
    metric     a big number with its label
    selection  chips
    follow-ups chips under the reply

A click sends a normal chat message naming the product ("Tell me more about FOO-0043 Mineral Water
1L x12 in MUM"). The list the user clicked in is already the session's selection (4.2), so "the 2nd
one" works too; the SKU in the message just makes the router's and agent's job easy.

Experiments (the chat functions, no server):
  1. "Show low stock items in Mumbai"   -> reply + a table, saved with the message
  2. Click row 2                         -> the next turn asks about exactly that product
  3. "thanks!"                           -> no lookup, no blocks
"""

import json
import time

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy import text as sql

from app.db import SessionLocal
from app.llm import ROUTER_MODEL, client, types
from app.models import ChatMessage
from lessons.common import header
from lessons.l14_tracing import Tracer
from lessons.l41_history import load_messages, new_session
from lessons.l45_selection import from_trace
from lessons.l52_dispatch import pending, simple_read
from lessons.l62_verify import numbers_in
from lessons.l73_investigation import HANDLERS_V2
from lessons.l81_jobs import ensure_tables as ensure_job_tables
from lessons.l81_jobs import start_turn
from lessons.l91_response import BadPointer, Block, BlockSpec, build, ungrounded

ASHA = 1


def ensure_tables():
    ensure_job_tables()
    with SessionLocal() as db:  # chat_messages existed before 9.2: add the display column
        db.execute(sql("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS display json"))
        db.commit()


# --- The layout step --------------------------------------------------------------------------------

class Layout(BaseModel):
    blocks: list[BlockSpec] = Field(default_factory=list, max_length=3)
    follow_ups: list[str] = Field(default_factory=list, max_length=3, description=(
        "Up to 3 short questions the user may ask next, answerable with the warehouse tools."))


class Display(BaseModel):
    blocks: list[Block] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


LAYOUT_PROMPT = (
    "You lay out a warehouse assistant's reply in a chat UI. You get the question, the reply, and the tool "
    "results it was based on, each with a call_id. Choose blocks that POINT at those results; code copies the "
    "data, you never write it. table: any list of rows the reply talks about (rows with a sku become clickable); "
    "pick the useful columns. metric: one key number worth highlighting. selection: products to choose from, "
    "only when there is no table of them. No blocks for a list the reply didn't use. Titles have no numbers. "
    "follow_ups: name only products and invoices that appear in the tool results."
)
LISTS = ("\n\nThe app shows lists from your tool results as a table under your reply. Don't repeat the list: "
         "say in 1-3 sentences what matters in it.")


def results_text(log: list) -> str:
    return "\n".join(f"[call_id {i}] {name}({json.dumps(dict(args))}) -> {json.dumps(result, default=str)[:3000]}"
                     for i, (name, args, result) in enumerate(log, 1))


def lay_out(question: str, reply: str, log: list, tracer=None) -> Display:
    if not log:  # nothing was looked up: nothing to point at
        return Display()
    t0 = time.perf_counter()
    r = client.models.generate_content(
        model=ROUTER_MODEL,
        contents=f"Question: {question}\n\nReply: {reply}\n\nTool results:\n{results_text(log)}",
        config=types.GenerateContentConfig(system_instruction=LAYOUT_PROMPT, temperature=0,
                                           response_mime_type="application/json", response_schema=Layout),
    )
    if tracer:
        tracer.model_call("layout", r, time.perf_counter() - t0)
    layout = Layout.model_validate_json(r.text)
    blocks = []
    for spec in layout.blocks:
        try:
            if numbers_in(spec.title):
                raise BadPointer(f"title {spec.title!r} has a number")
            blocks.append(build(spec, log))
        except BadPointer as e:
            print(f"   [layout] dropped a {spec.kind}: {e}")  # extra, not essential: drop, don't retry
    follow_ups = [q for q in layout.follow_ups if not ungrounded(q, log)]
    return Display(blocks=blocks, follow_ups=follow_ups)


def save_display(session_id: int, display: Display):
    """Attach the display to this session's newest assistant message (the reply just saved)."""
    with SessionLocal() as db:
        m = db.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id,
                                                 ChatMessage.role == "assistant")
                       .order_by(ChatMessage.id.desc()).limit(1)).first()
        if m:
            m.display = display.model_dump(mode="json")
            db.commit()


def displayed(handler):
    """Wrap a quick-path handler: same reply, plus blocks built from its tool log."""
    def run(user_id, session_id, text, d, tracer=None, log=None, **extra):
        log = [] if log is None else log
        reply = handler(user_id, session_id, text, d, tracer=tracer, log=log, extra_prompt=LISTS, **extra)
        display = lay_out(text, reply, log, tracer)
        if display.blocks or display.follow_ups:
            save_display(session_id, display)
        if on_display := getattr(tracer, "on_display", None):  # 8.3's StreamTracer: send it to the browser
            on_display(display)
        return reply
    return run


HANDLERS_V3 = {**HANDLERS_V2, "simple_read": displayed(simple_read), "pending": displayed(pending)}


# --- Experiments ---------------------------------------------------------------------------------

class NoJobs:
    def add_task(self, fn, *args):
        raise RuntimeError("these experiments don't start investigations")


def turn(sid: int, text: str) -> tuple[str, dict | None, list[str]]:
    """One chat turn -> (reply, the display saved with it, the tool calls made)."""
    t = Tracer(text)
    reply = start_turn(ASHA, sid, text, NoJobs(), tracer=t, handlers=HANDLERS_V3)["reply"]
    return reply, load_messages(sid)[-1]["display"], from_trace(t)[1]


def show(reply: str, display: dict | None, calls: list[str]):
    print(f"\n   tool calls: {calls}\n   REPLY: {reply}")
    for b in (display or {}).get("blocks", []):
        if b["type"] == "table":
            print(f"\n   [table] {b['title']}  ({b['source']}, clickable rows: {len(b['options'])})")
            print("     " + " | ".join(b["columns"]))
            for row in b["rows"][:8]:
                print("     " + " | ".join(str(row[c]) for c in b["columns"]))
        elif b["type"] == "metric":
            print(f"\n   [metric] {b['label']}: {b['value']}  ({b['source']})")
        else:
            print(f"\n   [selection] {b['title']}: {[o['sku'] for o in b['options']]}")
    print(f"\n   follow-ups: {(display or {}).get('follow_ups', [])}")


def click(option: dict) -> str:
    """What the UI sends when a row is clicked (same text as App.tsx's askAbout)."""
    where = f" in {option['warehouse']}" if option.get("warehouse") else ""
    return f"Tell me more about {option['sku']} {option.get('name', '')}{where}".strip()


def main():
    ensure_tables()
    sid = new_session(ASHA)

    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: 'Show low stock items in Mumbai'")
    reply, display, calls = turn(sid, "Show low stock items in Mumbai")
    show(reply, display, calls)

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: click row 2 of the table")
    table = next((b for b in (display or {}).get("blocks", []) if b["type"] == "table" and b["options"]), None)
    if table is None:
        print("   no clickable table in experiment 1")
    else:
        row2 = table["options"][1]
        message = click(row2)
        print(f"   row 2 = {row2}\n   the UI sends: {message!r}")
        reply, display2, calls = turn(sid, message)
        show(reply, display2, calls)
        print(f"\n   asked about {row2['sku']}: {'yes' if any(row2['sku'] in c for c in calls) else 'NO'}")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: 'thanks!'")
    reply, display, calls = turn(sid, "thanks!")
    show(reply, display, calls)


if __name__ == "__main__":
    main()
