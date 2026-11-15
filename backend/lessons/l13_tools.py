"""
Lesson 1.3: Multiple tools and chaining
========================================

Run from backend/:   uv run python -m lessons.l13_tools

In 1.1/1.2 the SKU was hard-coded in the tool description. That doesn't scale.
Now the model gets FOUR tools and has to find IDs itself:

    search_products(query)                        -> find the SKU from words
    get_stock(sku, warehouse_code)                -> current quantity
    list_movements(sku, warehouse_code, from, to) -> every stock change (the history)
    get_invoice(number)                           -> who bought, how much, paid or not

The loop is the SAME as 1.2. We never write "search first, then movements".
The model chains the tools itself, using the output of one as input to the next.

Experiments:
  1. Two-hop chain: words -> search_products -> get_stock
  2. Investigation: "why is padlock stock in BLR dropping?" (a planted story)
  3. Three-hop chain: product -> movements -> invoice
"""

from datetime import date

from app.db import query
from app.llm import types
from lessons.common import header
from lessons.l12_agent_loop import get_stock, run_agent  # the tool + the loop from 1.2, reused as-is

MAX_STEPS = 8
TODAY = date.today()


# --- New tools (get_stock comes from 1.2) ----------------------------------

def search_products(query_text: str) -> dict:
    # Every word must appear in the name or SKU: "brass lock" -> '%brass%' AND '%lock%'
    words = query_text.lower().split()
    where = " AND ".join(f"(lower(name) LIKE :w{i} OR lower(sku) LIKE :w{i})" for i in range(len(words)))
    rows = query(
        f"SELECT sku, name, unit, reorder_level FROM products WHERE {where} ORDER BY sku LIMIT 10",
        **{f"w{i}": f"%{w}%" for i, w in enumerate(words)},
    )
    return {"matches": rows} if rows else {"matches": [], "hint": "no product matched; try fewer or simpler words"}


def list_movements(sku: str, warehouse_code: str, date_from: str | None = None, date_to: str | None = None) -> dict:
    rows = query(
        """SELECT m.occurred_at::date AS date, m.movement_type AS type, m.quantity,
                  m.reference, m.note
           FROM stock_movements m
           JOIN products p ON p.id = m.product_id
           JOIN warehouses w ON w.id = m.warehouse_id
           WHERE p.sku = :sku AND w.code = :wh
             AND (CAST(:dfrom AS date) IS NULL OR m.occurred_at::date >= CAST(:dfrom AS date))
             AND (CAST(:dto AS date) IS NULL OR m.occurred_at::date <= CAST(:dto AS date))
           ORDER BY m.occurred_at
           LIMIT 200""",
        sku=sku, wh=warehouse_code, dfrom=date_from, dto=date_to,
    )
    return {"sku": sku, "warehouse": warehouse_code, "count": len(rows), "movements": rows}


def get_invoice(number: str) -> dict:
    head = query(
        """SELECT i.number, c.name AS customer, w.code AS warehouse, i.issue_date, i.due_date,
                  i.status, i.total,
                  COALESCE((SELECT SUM(amount) FROM payments WHERE invoice_id = i.id), 0) AS paid
           FROM invoices i JOIN customers c ON c.id = i.customer_id JOIN warehouses w ON w.id = i.warehouse_id
           WHERE i.number = :n""",
        n=number,
    )
    if not head:
        return {"error": f"no invoice {number!r}"}
    lines = query(
        """SELECT p.sku, p.name, l.quantity, l.unit_price, l.line_total
           FROM invoice_lines l JOIN products p ON p.id = l.product_id
           JOIN invoices i ON i.id = l.invoice_id WHERE i.number = :n""",
        n=number,
    )
    return {**head[0], "lines": lines}


TOOL_REGISTRY = {
    "search_products": search_products,
    "get_stock": get_stock,
    "list_movements": list_movements,
    "get_invoice": get_invoice,
}

# --- Declarations: what the model reads. Notice: NO hard-coded SKUs anymore. ---

WAREHOUSE = {
    "type": "string",
    "enum": ["BLR", "MUM", "DEL"],
    "description": "BLR = Bengaluru/Bangalore, MUM = Mumbai/Bombay, DEL = Delhi.",
}
SKU = {"type": "string", "description": "Exact SKU like HAR-0018. Get it from search_products; never guess."}

DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_products",
        description="Find products by name words. Use this FIRST whenever the user names a product, to get its SKU.",
        parameters_json_schema={
            "type": "object",
            "properties": {"query_text": {"type": "string", "description": "Product words, e.g. 'brass padlock'."}},
            "required": ["query_text"],
        },
    ),
    types.FunctionDeclaration(
        name="get_stock",
        description="Current on-hand quantity of one product in one warehouse.",
        parameters_json_schema={
            "type": "object",
            "properties": {"sku": SKU, "warehouse_code": WAREHOUSE},
            "required": ["sku", "warehouse_code"],
        },
    ),
    types.FunctionDeclaration(
        name="list_movements",
        description=(
            "Every stock change for one product in one warehouse, oldest first. quantity is signed "
            "(+ in, - out). Types: purchase_in, sale_out, transfer_in, transfer_out, adjustment, return_in. "
            "reference links to the source document (INV-... = invoice, PO-... = purchase order). "
            "Use this to explain WHY stock changed."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "sku": SKU,
                "warehouse_code": WAREHOUSE,
                "date_from": {"type": "string", "description": "Optional start date, YYYY-MM-DD."},
                "date_to": {"type": "string", "description": "Optional end date, YYYY-MM-DD."},
            },
            "required": ["sku", "warehouse_code"],
        },
    ),
    types.FunctionDeclaration(
        name="get_invoice",
        description="One invoice by number (e.g. INV-00012): customer, lines, total, amount paid, status.",
        parameters_json_schema={
            "type": "object",
            "properties": {"number": {"type": "string", "description": "Invoice number like INV-00012."}},
            "required": ["number"],
        },
    ),
]

CONFIG = types.GenerateContentConfig(
    system_instruction=(
        f"Today is {TODAY.isoformat()}. You are a warehouse assistant. "
        "Use tools for every fact; never guess numbers or SKUs. "
        "When you explain something, cite the evidence (dates, quantities, references)."
    ),
    tools=[types.Tool(function_declarations=DECLARATIONS)],
)


def ask(question: str, tracer=None) -> str:
    """1.2's loop, with this lesson's tools and system prompt."""
    return run_agent(question, config=CONFIG, registry=TOOL_REGISTRY, max_steps=MAX_STEPS, tracer=tracer)


def main():
    # --- Experiment 1 -------------------------------------------------------------
    header("Experiment 1: two-hop chain (words -> SKU -> stock)")
    print(ask("How many brass locks do we have in Bangalore?"))
    print("\n-> Nobody told it HAR-0018. It found the SKU with search_products, then")
    print("   copied it into get_stock. The output of tool 1 became the input of tool 2.")

    # --- Experiment 2 -------------------------------------------------------------
    header("Experiment 2: investigation (a planted story)")
    print(ask("Why is padlock stock in BLR dropping?"))

    truth = query(
        """SELECT m.movement_type, COUNT(*) AS n, SUM(m.quantity) AS total,
                  COUNT(*) FILTER (WHERE m.reference IS NULL AND m.note IS NULL) AS unexplained
           FROM stock_movements m JOIN products p ON p.id = m.product_id
           JOIN warehouses w ON w.id = m.warehouse_id
           WHERE p.sku = 'HAR-0018' AND w.code = 'BLR' GROUP BY 1 ORDER BY 1"""
    )
    print("\n   DB truth (no AI):")
    for r in truth:
        print(f"     {r['movement_type']:<12} {r['n']:>3} moves, total {r['total']:>5}, no reference/note: {r['unexplained']}")
    print("-> Did the agent spot the adjustments with NO reference and NO note?")

    # --- Experiment 3 -------------------------------------------------------------
    header("Experiment 3: three-hop chain (product -> movements -> invoice)")
    print(ask("Who bought padlocks from BLR most recently, and have they paid for it?"))


if __name__ == "__main__":
    main()
