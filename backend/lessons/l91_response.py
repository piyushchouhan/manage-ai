"""
Lesson 9.1: Response schema
============================

Run from backend/:   uv run python -m lessons.l91_response

Until now every reply was one markdown string. A list of 9 low-stock items came back as text the
model RETYPED from the tool result: it can drop a row, round a number, reorder the list. And the UI
can't do anything with text: no sorting, no clicking "the 2nd one".

The response becomes data:

    {summary:    "Mumbai has 5 products below reorder level; mineral water is out.",   <- model, checked (6.2/6.3)
     blocks: [
       {type: table,     title, columns, rows,  source: "list_low_stock · call #1"},  <- CODE copies the rows
       {type: selection, title, options: [{sku, name, warehouse}], source},            <- CODE, for clicking (9.2)
       {type: metric,    label, value,          source: "get_stock · call #2"},       <- CODE reads the value
     ],
     follow_ups: ["Movements for FOO-0043 in MUM?", ...],
     verified: true}

The key rule: the model never writes table rows or metric values. It only POINTS at a tool result,
the same way 6.2's facts do:

    model writes  {kind: "table", title: "Low stock in Mumbai", call_id: 1, path: "items",
                   columns: ["sku", "name", "quantity", "reorder_level"]}
    code builds   rows = log[call 1].result["items"], only those columns          (build)
    a rule checks the pointer is real: call happened, path exists, columns exist   (blocks_resolve)

So a table can't contain a number the database didn't return. It goes through 6.3's retry loop like
every other rule: a bad pointer -> one retry -> safe failure.

Experiments:
  1. "low stock in Mumbai"            -> table + selection
  2. "padlocks in BLR and DEL"        -> metrics
  3. "who owes us the most"           -> a table with the columns the model chose
  4. Broken pointers (no model call)   -> each caught by a rule
     (incl. a follow-up naming INV-00012, which the first run of experiment 3 really did)
"""

import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.llm import types
from lessons.common import header
from lessons.l42_selections import as_selection
from lessons.l61_typed_tools import DECLS_V7
from lessons.l62_verify import lookup, numbers_in, same
from lessons.l63_rules import (CONFIG_V9, FINAL_DECL_V9, RULES, AnswerV9, Context, answer_checked, check,
                               safe_failure)

ASHA = 1
MAX_ROWS = 25


# --- What the MODEL writes: pointers, not data ----------------------------------------------------

class BlockSpec(BaseModel):
    kind: Literal["table", "selection", "metric"] = Field(description=(
        "table = a list to scan. selection = products the user may pick next (rows must have a sku). "
        "metric = one key number."))
    title: str = Field(description="A short title, with no numbers in it.")
    call_id: int = Field(ge=1, description="call_id of the tool result to show.")
    path: str = Field(description=(
        "table/selection: path to a LIST of rows in that result, e.g. 'items', 'customers', 'suppliers'. "
        "metric: path to ONE value, e.g. 'quantity', 'summary_by_type.adjustment.total_qty'."))
    columns: list[str] = Field(default_factory=list, description=(
        "table only: the keys of each row to show, in order. Pick the useful ones; empty = all."))


class Presented(AnswerV9):
    """6.3's answer (summary, facts, computed) + how to show it."""
    blocks: list[BlockSpec] = Field(default_factory=list, max_length=4)
    follow_ups: list[str] = Field(default_factory=list, max_length=3, description=(
        "Up to 3 short questions the user may want to ask next, that the tools can answer."))


PRESENT = (
    "\n\nHow to present the answer: the summary is 1-3 sentences. Do NOT copy lists into the summary: show "
    "them as blocks that point at the tool result (call_id + path); code fills in the rows and values. "
    "Use a table for a list, a selection for products the user may want to pick next (any list with SKUs), "
    "a metric for a single key number."
)
FINAL_DECL_V10 = types.FunctionDeclaration(
    name="final_answer",
    description=FINAL_DECL_V9.description + " Lists and key numbers go in blocks, pointing at tool results.",
    parameters_json_schema=Presented.model_json_schema(),
)
CONFIG_V10 = CONFIG_V9.model_copy(update={
    "tools": [types.Tool(function_declarations=[*DECLS_V7.values(), FINAL_DECL_V10])],
    "system_instruction": CONFIG_V9.system_instruction + PRESENT,
})


# --- What the UI gets: data, built by code ----------------------------------------------------------

class Table(BaseModel):
    type: Literal["table"] = "table"
    title: str
    columns: list[str]
    rows: list[dict]
    source: str  # "list_low_stock · call #1": where every row came from
    options: list[dict] = Field(default_factory=list)  # 9.2: rows with a sku -> what clicking row i refers to


class Metric(BaseModel):
    type: Literal["metric"] = "metric"
    label: str
    value: float | int | str
    source: str


class Selection(BaseModel):
    type: Literal["selection"] = "selection"
    title: str
    options: list[dict]  # [{sku, name, warehouse}]: what a click sends back (4.2's selection)
    source: str


Block = Annotated[Table | Metric | Selection, Field(discriminator="type")]


class Response(BaseModel):
    summary: str
    blocks: list[Block] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    verified: bool


class BadPointer(Exception):
    pass


def build(spec: BlockSpec, log: list) -> Table | Metric | Selection:
    """Follow the model's pointer into the tool log and copy the data out. Raises BadPointer."""
    if not 1 <= spec.call_id <= len(log):
        raise BadPointer(f"call #{spec.call_id} never happened")
    name, _, result = log[spec.call_id - 1]
    source = f"{name} · call #{spec.call_id}"
    try:
        data = lookup(result, spec.path)
    except (KeyError, IndexError, TypeError, ValueError):
        raise BadPointer(f"{name}'s result has no {spec.path!r}")

    if spec.kind == "metric":
        if data is None or isinstance(data, (dict, list)):
            raise BadPointer(f"{spec.path!r} is not a single value")
        return Metric(label=spec.title, value=data, source=source)

    if not (isinstance(data, list) and data and all(isinstance(r, dict) for r in data)):
        raise BadPointer(f"{spec.path!r} is not a list of rows")
    if spec.kind == "selection":
        if "sku" not in data[0]:
            raise BadPointer("a selection needs rows with a sku (it's what a click sends back)")
        return Selection(title=spec.title, options=as_selection(data), source=source)

    columns = spec.columns or list(data[0])
    if missing := [c for c in columns if c not in data[0]]:
        raise BadPointer(f"{name} rows have no {missing}; they have {list(data[0])}")
    rows = data[:MAX_ROWS]
    return Table(title=spec.title, columns=columns, source=source, rows=[{c: r.get(c) for c in columns} for r in rows],
                 options=as_selection(rows) if "sku" in rows[0] else [])  # clickable even if sku isn't a column


# --- Two more rules for 6.3's loop ------------------------------------------------------------------

def blocks_resolve(a: Presented, log: list, c: Context) -> list[str]:
    out = []
    for i, spec in enumerate(a.blocks, 1):
        try:
            build(spec, log)
        except BadPointer as e:
            out.append(f"block {i} ({spec.kind} '{spec.title}'): {e}")
    return out


def titles_have_no_numbers(a: Presented, log: list, c: Context) -> list[str]:
    """A title is model text next to the data: '9 items low' could disagree with the 5 rows below it."""
    backed = [f.value for f in a.facts] + [x.value for x in a.computed]
    return [f"block title '{b.title}' has the number {n:g}, which is not a checked fact"
            for b in a.blocks for n in numbers_in(b.title) if not any(same(n, v) for v in backed)]


REFS = r"INV-\d{5}|[A-Z]{3}-\d{4}"  # invoice numbers and SKUs


def ungrounded(text: str, log: list) -> list[str]:
    """Invoice numbers / SKUs in the text that no tool returned."""
    seen = json.dumps([result for _, _, result in log], default=str)
    return [ref for ref in re.findall(REFS, text) if ref not in seen]


def follow_ups_grounded(a: Presented, log: list, c: Context) -> list[str]:
    """A follow-up is a button the user may click: it must not name an invoice or SKU nobody looked up."""
    return [f"follow-up '{q}' names {ref}, which no tool returned" for q in a.follow_ups for ref in ungrounded(q, log)]


RULES_V10 = {**RULES, "blocks point at real data": blocks_resolve, "titles have no numbers": titles_have_no_numbers,
             "follow-ups name only real things": follow_ups_grounded}


# --- The whole thing ------------------------------------------------------------------------------

def present(a: AnswerV9 | None, log: list, c: Context, ok: bool) -> Response:
    if not ok or not isinstance(a, Presented):
        return Response(summary=safe_failure(a, log, c), verified=False)  # 6.3: no blocks from a failed answer
    return Response(summary=a.summary, blocks=[build(b, log) for b in a.blocks], follow_ups=a.follow_ups,
                    verified=True)


def answer_presented(question: str, user_id: int):
    """-> (the model's answer, the tool log, context, passed?): 6.3's loop with 9.1's schema and rules."""
    return answer_checked(question, user_id, schema=Presented, config=CONFIG_V10, rules=RULES_V10)


def respond(question: str, user_id: int) -> Response:
    return present(*answer_presented(question, user_id))


# --- Experiments ---------------------------------------------------------------------------------

def show(r: Response):
    print(f"\n   RESPONSE (verified={r.verified})\n   summary: {r.summary}")
    for b in r.blocks:
        if isinstance(b, Table):
            print(f"\n   [table] {b.title}   ({b.source}, {len(b.rows)} rows)")
            print("     " + " | ".join(b.columns))
            for row in b.rows[:8]:
                print("     " + " | ".join(str(row[col]) for col in b.columns))
        elif isinstance(b, Metric):
            print(f"\n   [metric] {b.label}: {b.value}   ({b.source})")
        else:
            print(f"\n   [selection] {b.title}   ({b.source})")
            print("     " + "  ".join(f"[{i}] {o['sku']} {o.get('name', '')}" for i, o in enumerate(b.options, 1)))
    if r.follow_ups:
        print(f"\n   follow-ups: {r.follow_ups}")
    print(f"\n   JSON for the UI: {len(r.model_dump_json())} chars")


def broken(label: str, a: Presented, log: list, c: Context, **change):
    problems = check(a.model_copy(update=change), log, c, RULES_V10)
    print(f"\n   {label}: {'PASS' if not problems else 'CAUGHT'}")
    for p in problems:
        print(f"     XX {p}")


def main():
    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: 'Show low stock items in Mumbai'")
    a, log, c, ok = answer_presented("Show low stock items in Mumbai", ASHA)
    show(present(a, log, c, ok))

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: 'How many padlocks are in Bangalore and Delhi?'")
    show(respond("How many padlocks are in Bangalore and Delhi?", ASHA))

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: 'Which customers owe us the most?'")
    show(respond("Which customers owe us the most?", ASHA))

    # --- Experiment 4 ---------------------------------------------------------
    header("Experiment 4: broken pointers in experiment 1's answer (code only)")
    if not (ok and isinstance(a, Presented) and a.blocks):
        print("   experiment 1 has no blocks to break")
        return
    spec = a.blocks[0]
    broken("the real answer", a, log, c)
    broken("a column the tool never returned", a, log, c,
           blocks=[spec.model_copy(update={"kind": "table", "columns": ["sku", "profit"]})])
    broken("a call that never happened", a, log, c, blocks=[spec.model_copy(update={"call_id": 99})])
    broken("a metric pointing at a whole list", a, log, c,
           blocks=[spec.model_copy(update={"kind": "metric", "path": "items"})])
    broken("a title with a made-up count", a, log, c,
           blocks=[spec.model_copy(update={"title": "The 12 items running low"})])
    broken("a follow-up about an invoice nobody looked up", a, log, c,
           follow_ups=["Can we see details for invoice INV-00012?"])


if __name__ == "__main__":
    main()
