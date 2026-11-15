"""
Lesson 2.2: Good tool design
=============================

Run from backend/:   uv run python -m lessons.l22_tool_design

The model only ever sees two things about a tool: its DESCRIPTION and its
RESULTS. Good tool design is making both easy for the model to use:

  1. Errors are DATA, not exceptions. A raised exception kills the whole
     agent. An error dict with a hint lets the model fix its own mistake.
  2. Results are COMPACT. Every byte you return is re-sent on every later
     model call (1.4: list_movements alone was 3,497 chars).
  3. Rows are CAPPED with a `limit`, and a summary comes first.
  4. Descriptions say WHEN to use the tool, what it RETURNS, and what errors MEAN.

Same loop (1.2), same tracer (1.4). Only the tools change.

Experiments:
  1. Wrong SKU with a tool that RAISES -> the agent crashes
  2. Wrong SKU with errors-as-data + hint -> the model searches and recovers
  3. Compact list_movements vs 1.3's: result size and tokens, side by side
"""

import json

from app.db import query
from app.llm import types
from lessons.common import header
from lessons.l12_agent_loop import get_stock, run_agent
from lessons.l13_tools import (
    CONFIG as CONFIG_V1,
    DECLARATIONS as DECLARATIONS_V1,
    MAX_STEPS,
    SKU,
    TOOL_REGISTRY as TOOLS_V1,
    WAREHOUSE,
    list_movements,
)
from lessons.l14_tracing import Tracer, show

NOT_FOUND_HINT = "Call search_products with the product name to find the right SKU, then retry."


# --- 1. A BAD tool: raises on bad input (what most normal Python code does) -----

def strict_get_stock(sku: str, warehouse_code: str) -> dict:
    result = get_stock(sku, warehouse_code)
    if "error" in result:
        raise ValueError(f"SKU not found: {sku}")  # fine for a script, fatal for an agent
    return result


# --- 2. GOOD tools: errors as data, with a hint about what to do next ------------

def sku_exists(sku: str) -> bool:
    return bool(query("SELECT 1 FROM products WHERE sku = :sku", sku=sku))


def get_stock_v2(sku: str, warehouse_code: str) -> dict:
    result = get_stock(sku, warehouse_code)  # 1.2's tool
    if "error" in result:
        return {"error": "unknown_sku", "sku": sku, "hint": NOT_FOUND_HINT}
    return result


def list_movements_v2(sku: str, warehouse_code: str, date_from: str | None = None,
                      date_to: str | None = None, limit: int = 15) -> dict:
    if not sku_exists(sku):
        # 1.3 returned {"count": 0} here, which LOOKS like "no movements": a silent lie.
        return {"error": "unknown_sku", "sku": sku, "hint": NOT_FOUND_HINT}

    rows = list_movements(sku, warehouse_code, date_from, date_to)["movements"]  # 1.3's tool
    limit = max(1, min(int(limit), 50))  # cap: the model can ask for more, but never unlimited

    # Summary first: most questions ("why is it dropping?") are answered by totals alone.
    by_type: dict[str, dict] = {}
    for r in rows:
        t = by_type.setdefault(r["type"], {"moves": 0, "total_qty": 0, "without_reference": 0})
        t["moves"] += 1
        t["total_qty"] += r["quantity"]
        t["without_reference"] += r["reference"] is None

    # Compact rows: newest first, short keys, drop empty fields
    recent = [
        {k: v for k, v in (("d", r["date"]), ("t", r["type"]), ("q", r["quantity"]),
                           ("ref", r["reference"]), ("note", r["note"])) if v is not None}
        for r in reversed(rows[-limit:])
    ]
    return {
        "sku": sku, "warehouse": warehouse_code,
        "period": [rows[0]["date"], rows[-1]["date"]] if rows else None,
        "summary_by_type": by_type,
        "recent": recent,
        "rows_shown": len(recent), "rows_total": len(rows),  # tells the model there is more
        "keys": "d=date t=type q=signed quantity ref=document",
    }


def safe(fn):
    """Safety net for EVERY tool: an unexpected exception becomes data, not a crash."""
    def wrapper(**kwargs):
        try:
            return fn(**kwargs)
        except Exception as e:
            return {"error": type(e).__name__, "message": str(e)}
    return wrapper


TOOLS_V2 = {name: safe(fn) for name, fn in {
    **TOOLS_V1,  # search_products, get_invoice: unchanged from 1.3
    "get_stock": get_stock_v2,
    "list_movements": list_movements_v2,
}.items()}


# --- Better descriptions: WHEN to use, what it RETURNS, what errors MEAN ----------

DECLS_V2 = {d.name: d for d in DECLARATIONS_V1}  # start from 1.3, replace two
DECLS_V2["get_stock"] = types.FunctionDeclaration(
    name="get_stock",
    description=(
        "Current on-hand quantity of ONE product in ONE warehouse. "
        "Returns {sku, name, warehouse, quantity}. "
        "If it returns error 'unknown_sku', follow the hint: search, then retry with the right SKU."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {"sku": SKU, "warehouse_code": WAREHOUSE},
        "required": ["sku", "warehouse_code"],
    },
)
DECLS_V2["list_movements"] = types.FunctionDeclaration(
    name="list_movements",
    description=(
        "Stock history of ONE product in ONE warehouse. Use it to explain WHY stock changed. "
        "Returns summary_by_type (moves, total_qty, without_reference per movement type) for the whole "
        "period, plus the most recent rows (newest first). Read the summary first; ask for a bigger "
        "`limit` or a date range only if you need individual rows. A movement without a reference has "
        "no source document (unexplained). Quantity is signed: + in, - out."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "sku": SKU,
            "warehouse_code": WAREHOUSE,
            "date_from": {"type": "string", "description": "Optional start date, YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "Optional end date, YYYY-MM-DD."},
            "limit": {"type": "integer", "description": "Recent rows to return, 1-50. Default 15."},
        },
        "required": ["sku", "warehouse_code"],
    },
)
CONFIG_V2 = CONFIG_V1.model_copy(update={
    "tools": [types.Tool(function_declarations=list(DECLS_V2.values()))]
})


def ask(question: str, config, registry, tracer=None) -> str:
    return run_agent(question, config=config, registry=registry, max_steps=MAX_STEPS, tracer=tracer)


def main():
    wrong_sku_q = "What's the stock of SKU PAD-0050 in BLR?"  # real SKU is HAR-0018

    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: a tool that RAISES on a wrong SKU")
    try:
        ask(wrong_sku_q, CONFIG_V1, {**TOOLS_V1, "get_stock": strict_get_stock})
    except ValueError as e:
        print(f"\n   CRASH: ValueError: {e}")
        print("   -> The whole agent died on step 1. The model never got a chance to fix it.")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: errors as DATA, with a hint")
    print(ask(wrong_sku_q, CONFIG_V2, TOOLS_V2))
    print("\n-> The error went BACK to the model as a tool result. It read the hint,")
    print("   searched, and retried. Errors are information, not the end.")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: compact results (same question, 1.3 tools vs 2.2 tools)")
    v1 = json.dumps(list_movements("HAR-0018", "BLR"))
    v2 = json.dumps(list_movements_v2("HAR-0018", "BLR"))
    print(f"   list_movements result size:  1.3 = {len(v1):,} chars   2.2 = {len(v2):,} chars")

    question = "Why is padlock stock in BLR dropping?"
    traces = {}
    for label, config, tools in [("1.3 tools", CONFIG_V1, TOOLS_V1), ("2.2 tools", CONFIG_V2, TOOLS_V2)]:
        header(f"Experiment 3 - {label}")
        tracer = Tracer(f"{label} {question}")
        print(ask(question, config, tools, tracer=tracer)[:600], "...")
        traces[label] = tracer.path
    for label, path in traces.items():
        show(path)
    print("\n-> Compare total tokens and the prompt size at the final step.")
    print("   Did the 2.2 answer still find the unexplained adjustments?")


if __name__ == "__main__":
    main()
