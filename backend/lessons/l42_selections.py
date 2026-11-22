"""
Lesson 4.2: Selections and pending choices
===========================================

Run from backend/:   uv run python -m lessons.l42_selections

"show low stock items" -> a numbered list. Then "movements for the 2nd one".
Which one is "the 2nd one"? The model's old answer is only text in the history.
It may have been reworded, shortened ("...and 3 more") or reordered.

So CODE remembers what the user is looking at, per chat session (table chat_state):
    selection: the last list a tool returned, in order   [{sku, name, warehouse}, ...]
    pending:   a question the agent asked and is waiting on (3.3's Clarification)

Each turn:
    1. load the state from the DB
    2. put it in the system prompt as a numbered list ("2nd one" = item 2 of THAT list)
    3. run the agent; record every tool result (2.3's recording())
    4. save the new list if a tool returned one; clear pending once it's answered

Experiments:
  1. "show low stock items in Mumbai" -> the list is saved
  2. "movements for the 2nd one"      -> uses exactly item 2 of the saved list
  3. "stock of all of them in Delhi"  -> uses every item of the list
  4. Pending choice: "stock of box in Delhi?" -> agent asks -> user replies just "2"
"""

from app.db import SessionLocal, engine, query
from app.llm import types
from app.models import Base, ChatState
from lessons.common import header
from lessons.l13_tools import WAREHOUSE
from lessons.l22_tool_design import safe
from lessons.l23_injection import recording
from lessons.l33_clarify import CONFIG_V5, DECLS_V5, TOOLS_V5, Clarification
from lessons.l41_history import (as_reply, ensure_tables as ensure_chat_tables, load_messages,
                                 new_session, run_turn, save_turn, to_history)


def ensure_tables():
    ensure_chat_tables()
    Base.metadata.create_all(engine, tables=[ChatState.__table__])


# --- A tool that returns a LIST (so there is something to point at) --------------

def list_low_stock(warehouse_code: str | None = None) -> dict:
    rows = query(
        """SELECT p.sku, p.name, w.code AS warehouse, s.quantity, p.reorder_level
           FROM stock_levels s JOIN products p ON p.id = s.product_id
           JOIN warehouses w ON w.id = s.warehouse_id
           WHERE s.quantity < p.reorder_level
             AND (CAST(:wh AS text) IS NULL OR w.code = :wh)
           ORDER BY s.quantity - p.reorder_level, p.sku""",  # biggest shortage first
        wh=warehouse_code,
    )
    return {"count": len(rows), "items": rows}


LOW_STOCK_DECL = types.FunctionDeclaration(
    name="list_low_stock",
    description="Products whose on-hand quantity is below their reorder level, biggest shortage first.",
    parameters_json_schema={
        "type": "object",
        "properties": {"warehouse_code": {**WAREHOUSE, "description": WAREHOUSE["description"] + " Omit for all."}},
    },
)

TOOLS_V6 = {**TOOLS_V5, "list_low_stock": safe(list_low_stock)}
DECLS_V6 = {**DECLS_V5, "list_low_stock": LOW_STOCK_DECL}
CONFIG_V6 = CONFIG_V5.model_copy(update={
    "tools": [types.Tool(function_declarations=list(DECLS_V6.values()))],
    "system_instruction": CONFIG_V5.system_instruction
    + " When you show several products, number them in the exact order the tool returned them.",
})


# --- The state: load, save, and turn into prompt text ----------------------------------

def load_state(session_id: int) -> dict:
    with SessionLocal() as db:
        s = db.get(ChatState, session_id)
        return {"selection": s.selection, "pending": s.pending} if s else {"selection": [], "pending": None}


def save_state(session_id: int, state: dict):
    with SessionLocal() as db:
        db.merge(ChatState(session_id=session_id, **state))  # insert or update
        db.commit()


def last_list(log: list) -> list[dict] | None:
    """The last list of products any tool returned this turn (low stock items, search options)."""
    for _name, _args, result in reversed(log):
        for key in ("items", "options"):
            rows = result.get(key) if isinstance(result, dict) else None
            if rows and "sku" in rows[0]:
                return [{k: r[k] for k in ("sku", "name", "warehouse") if k in r} for r in rows]
    return None


def state_prompt(state: dict) -> str:
    lines = []
    if state["selection"]:
        lines.append("\n\nThe user is looking at this numbered list. 'the 2nd one', 'all of them', "
                     "'those' refer to it:")
        for i, item in enumerate(state["selection"], 1):
            where = f" ({item['warehouse']})" if "warehouse" in item else ""
            lines.append(f"{i}. {item['sku']} {item['name']}{where}")
    if state["pending"]:
        p = state["pending"]
        lines.append(f"\nYou asked the user: {p['asked']!r} about their question {p['question']!r}. "
                     "Their message is probably the choice: pick from the list, then answer the ORIGINAL question.")
    return "\n".join(lines)


# --- One chat turn: 4.1's turn + state ------------------------------------------------

def chat(session_id: int, text: str, tracer=None, log: list | None = None,
         history: list[types.Content] | None = None, extra_prompt: str = "", prepare=None,
         tools: list[str] | None = None, extra_tools: dict | None = None, scope=None,
         registry: dict | None = None, decls: dict | None = None) -> str:
    """log: pass a list to see every tool call afterwards (the experiments check SKUs with it).
    history / extra_prompt: 4.3 passes a compacted history and the running summary; default = full history.
    prepare: 4.4's budget check, run before every model call.
    tools: 4.5 sends only these tool declarations (names from DECLS_V6); default = all.
    extra_tools: 4.6 adds per-user tools, {name: (function, declaration)}, always sent.
    scope: 4.7 wraps EVERY tool, scope(fn) -> fn, so results are limited to the user's warehouses.
    registry / decls: the tool set, default this lesson's V6 (4.7 passes 6.1's typed V7)."""
    registry, decls = registry or TOOLS_V6, decls or DECLS_V6
    extra_tools = extra_tools or {}
    state = load_state(session_id)
    config = CONFIG_V6.model_copy(update={
        "system_instruction": CONFIG_V6.system_instruction + extra_prompt + state_prompt(state),
        "tools": [types.Tool(function_declarations=[decls[n] for n in (tools or decls)]
                                                   + [decl for _, decl in extra_tools.values()])],
    })
    log = [] if log is None else log
    if history is None:
        history = to_history(load_messages(session_id))
    result = run_turn(text, tracer=tracer, history=history,
                      config=config, prepare=prepare,
                      registry=recording({n: scope(fn) if scope else fn for n, fn in
                                          {**registry, **{n: fn for n, (fn, _) in extra_tools.items()}}.items()}, log))

    if isinstance(result, Clarification):  # the agent is waiting on the user
        state = {"selection": result.options,
                 "pending": {"question": text, "asked": result.question, "options": result.options}}
    else:
        state = {"selection": last_list(log) or state["selection"], "pending": None}
    reply = as_reply(result)
    save_turn(session_id, text, reply)
    save_state(session_id, state)
    return reply


def used(log: list, tool: str) -> list[str]:
    return [args.get("sku") for name, args, _ in log if name == tool]


def main():
    ensure_tables()
    sid = new_session()

    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: a list is shown -> code saves it")
    print(chat(sid, "show low stock items in Mumbai"))
    selection = load_state(sid)["selection"]
    print("\n   chat_state.selection (saved in the DB):")
    for i, item in enumerate(selection, 1):
        print(f"     {i}. {item['sku']}  {item['name']}")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: 'movements for the 2nd one'")
    calls: list = []
    print(chat(sid, "movements for the 2nd one", log=calls))
    print(f"\n   saved item 2 = {selection[1]['sku']}   agent used = {used(calls, 'list_movements')}")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: 'all of them'")
    calls = []
    print(chat(sid, "what's the stock of all of them in Delhi?", log=calls))
    want, got = {i["sku"] for i in selection}, set(used(calls, "get_stock"))
    print(f"\n   saved list = {sorted(want)}")
    print(f"   agent used = {sorted(got)}   {'all covered' if want <= got else 'MISSED ' + str(sorted(want - got))}")

    # --- Experiment 4 ---------------------------------------------------------
    header("Experiment 4: a pending choice, answered with just '2'")
    sid2 = new_session()
    print(chat(sid2, "What's the stock of box in Delhi?"))
    state = load_state(sid2)
    print(f"\n   chat_state.pending = {state['pending']}")
    calls = []
    print(chat(sid2, "2", log=calls))
    if state["pending"]:
        print(f"\n   option 2 = {state['pending']['options'][1]['sku']}   agent used = {used(calls, 'get_stock')}")
    print(f"   pending after the answer = {load_state(sid2)['pending']}")


if __name__ == "__main__":
    main()
