"""
Lesson 6.1: Typed tool I/O
===========================

Run from backend/:   uv run python -m lessons.l61_typed_tools

The model writes tool arguments as JSON. JSON can say anything: limit -5, date
"2026-02-31", warehouse "Bangalore". Until now the tools trusted it:

    limit -5            -> silently became 1 row        (the model never knew)
    date "2026-02-31"   -> Postgres error with SQL text  (leaks internals, confusing)

Typed I/O = a Pydantic model for every tool's ARGUMENTS and RESULT, checked by code:

    model's args --[Args model]--> tool --[Result model]--> back to the model
                    |                        |
          bad -> {"error": "invalid_arguments",      bad -> OUR bug: logged, and the model
                  "problems": [field, problem, got]}        gets "internal_error, don't retry"
                  the model reads it and fixes the call

One source of truth: the SAME Args model gives the JSON schema the model sees
(parameters_json_schema) and the validator code runs. They can't drift apart.

Harmless mistakes are fixed, not rejected: " har-0018 " -> "HAR-0018".
Meaningful ones are rejected with a reason: 200 rows, February 31st, "Bangalore".

Experiments:
  1. No model: the same bad arguments to the old tools (4.x) and the typed tools
  2. A buggy tool: the result check catches OUR mistake before the model sees it
  3. The agent makes bad calls and corrects itself from the error
"""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from app.db import query
from app.llm import types
from lessons.common import header
from lessons.l13_tools import SKU, WAREHOUSE
from lessons.l22_tool_design import safe
from lessons.l23_injection import recording
from lessons.l32_dates import DateSpec
from lessons.l33_clarify import ask
from lessons.l41_history import as_reply
from lessons.l42_selections import CONFIG_V6, DECLS_V6, TOOLS_V6


def upper(v):
    return v.strip().upper() if isinstance(v, str) else v


Sku = Annotated[str, BeforeValidator(upper), Field(pattern=r"^[A-Z]{3}-\d{4}$", description=SKU["description"])]
Warehouse = Annotated[Literal["BLR", "MUM", "DEL"], BeforeValidator(upper), Field(description=WAREHOUSE["description"])]


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an argument the tool doesn't have is an error, not ignored


# --- Arguments ----------------------------------------------------------------------------

class StockArgs(Args):
    sku: Sku
    warehouse_code: Warehouse


class MovementArgs(Args):
    sku: Sku
    warehouse_code: Warehouse
    date_from: date | None = Field(None, description="Optional start date, YYYY-MM-DD. Get it from resolve_period.")
    date_to: date | None = Field(None, description="Optional end date, YYYY-MM-DD. Get it from resolve_period.")
    limit: int = Field(15, ge=1, le=50, description="Recent rows to return, 1-50. Default 15.")

    @model_validator(mode="after")
    def dates_in_order(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError(f"date_from {self.date_from} is after date_to {self.date_to}")
        return self


class InvoiceArgs(Args):
    number: str = Field(pattern=r"^INV-\d{5}$", description="Invoice number like INV-00558.")


class SearchArgs(Args):
    query_text: str = Field(min_length=2, max_length=100, description="Product name as the user wrote it.")


class LowStockArgs(Args):
    warehouse_code: Warehouse | None = Field(None, description=WAREHOUSE["description"] + " Omit for all.")


# --- Results ----------------------------------------------------------------------------------

class StockResult(BaseModel):
    sku: str
    name: str
    warehouse: str
    quantity: int = Field(ge=0)


class MovementType(BaseModel):
    moves: int
    total_qty: int
    without_reference: int


class MovementsResult(BaseModel):
    sku: str
    warehouse: str
    period: list[date] | None
    summary_by_type: dict[str, MovementType]
    recent: list[dict]
    rows_shown: int
    rows_total: int
    keys: str


class InvoiceLine(BaseModel):
    sku: str
    name: str
    quantity: int
    unit_price: float
    line_total: float


class InvoiceResult(BaseModel):
    number: str
    customer: str
    warehouse: str
    issue_date: date
    due_date: date
    status: Literal["paid", "partially_paid", "unpaid", "cancelled"]
    total: float
    paid: float
    lines: list[InvoiceLine]


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="allow")  # product / options / matches depend on the status
    status: Literal["match", "ambiguous", "not_found"]


class LowStockItem(BaseModel):
    sku: str
    name: str
    warehouse: str
    quantity: int
    reorder_level: int


class LowStockResult(BaseModel):
    count: int
    items: list[LowStockItem]


class PeriodResult(BaseModel):
    date_from: date | None
    date_to: date | None


# --- Added in 7.2: the planner showed there was no way to FIND unpaid invoices ----------------

def list_unpaid_invoices(warehouse_code: str | None = None, overdue_only: bool = True, limit: int = 10) -> dict:
    """Open balances per customer AND warehouse (so 4.7's filter can drop warehouses a user can't see)."""
    rows = query(
        """WITH open AS (
             SELECT i.customer_id, i.warehouse_id, i.due_date,
                    i.total - COALESCE((SELECT SUM(amount) FROM payments p WHERE p.invoice_id = i.id), 0) AS balance
             FROM invoices i
             WHERE i.status IN ('unpaid', 'partially_paid')
               AND (NOT :overdue OR i.due_date < CURRENT_DATE))
           SELECT c.name AS customer, w.code AS warehouse, COUNT(*) AS open_invoices,
                  ROUND(SUM(o.balance)::numeric, 2) AS outstanding, MIN(o.due_date) AS oldest_due,
                  CURRENT_DATE - MIN(o.due_date) AS days_overdue, c.credit_limit,
                  (SELECT COALESCE(SUM(p.amount), 0) FROM payments p JOIN invoices i2 ON i2.id = p.invoice_id
                   WHERE i2.customer_id = c.id AND i2.warehouse_id = w.id) AS paid_ever
           FROM open o JOIN customers c ON c.id = o.customer_id JOIN warehouses w ON w.id = o.warehouse_id
           WHERE CAST(:wh AS text) IS NULL OR w.code = :wh
           GROUP BY c.id, c.name, w.id, w.code, c.credit_limit
           ORDER BY outstanding DESC LIMIT :limit""",
        overdue=overdue_only, wh=warehouse_code, limit=limit,
    )
    return {"count": len(rows), "customers": rows}


class UnpaidArgs(Args):
    warehouse_code: Warehouse | None = Field(None, description=WAREHOUSE["description"] + " Omit for all.")
    overdue_only: bool = Field(True, description="True = only invoices past their due date.")
    limit: int = Field(10, ge=1, le=25, description="Rows to return, biggest outstanding first. 1-25.")


class UnpaidRow(BaseModel):
    customer: str
    warehouse: str
    open_invoices: int
    outstanding: float
    oldest_due: date
    days_overdue: int
    credit_limit: float
    paid_ever: float


class UnpaidResult(BaseModel):
    count: int
    customers: list[UnpaidRow]


UNPAID_DESCRIPTION = (
    "Customers who owe money: open (unpaid or partly paid) invoices per customer and warehouse, biggest "
    "outstanding first. Each row: open_invoices, outstanding, oldest_due, days_overdue, credit_limit, and "
    "paid_ever (0 = this customer has never paid anything at this warehouse). Use it for questions about "
    "who owes us, late payers, cash flow, credit limits."
)


# --- Added in 7.3: nothing could see suppliers, so "why do things keep running low?" had no answer ---

def list_suppliers(warehouse_code: str | None = None) -> dict:
    """Per supplier AND warehouse (4.7's filter drops warehouses a user can't see)."""
    rows = query(
        """SELECT s.name AS supplier, w.code AS warehouse, s.lead_time_days,
                  COUNT(*) AS products,
                  COUNT(*) FILTER (WHERE sl.quantity < p.reorder_level) AS low_now,
                  COUNT(*) FILTER (WHERE sl.quantity = 0) AS out_now
           FROM suppliers s JOIN products p ON p.supplier_id = s.id
           JOIN stock_levels sl ON sl.product_id = p.id JOIN warehouses w ON w.id = sl.warehouse_id
           WHERE CAST(:wh AS text) IS NULL OR w.code = :wh
           GROUP BY s.id, s.name, s.lead_time_days, w.code
           ORDER BY s.lead_time_days DESC, low_now DESC, w.code""",
        wh=warehouse_code,
    )
    return {"count": len(rows), "suppliers": rows}


class SupplierArgs(Args):
    warehouse_code: Warehouse | None = Field(None, description=WAREHOUSE["description"] + " Omit for all.")


class SupplierRow(BaseModel):
    supplier: str
    warehouse: str
    lead_time_days: int = Field(ge=0)
    products: int
    low_now: int
    out_now: int


class SupplierResult(BaseModel):
    count: int
    suppliers: list[SupplierRow]


SUPPLIERS_DESCRIPTION = (
    "Suppliers and how their products are doing, per supplier and warehouse, slowest supplier first. Each row: "
    "lead_time_days (days from order to delivery), products (how many of its products this warehouse stocks), "
    "low_now (how many are below reorder level right now), out_now (how many are at 0). Use it for questions "
    "about suppliers, deliveries, lead times, or why products keep running low."
)


# --- The wrapper --------------------------------------------------------------------------------

def problems(e: ValidationError) -> list[dict]:
    """Pydantic's errors -> short, model-readable lines."""
    out = []
    for err in e.errors():
        got = err.get("input")
        out.append({"field": ".".join(map(str, err["loc"])) or "(arguments)", "problem": err["msg"],
                    "got": got if isinstance(got, (str, int, float, bool, type(None))) else str(got)})
    return out


def typed(fn, args_model: type[BaseModel], result_model: type[BaseModel]):
    """Check args before the tool runs, and the result before the model sees it."""
    def call(**raw):
        try:
            args = args_model.model_validate(raw)
        except ValidationError as e:
            return {"error": "invalid_arguments", "problems": problems(e),
                    "hint": "Nothing was run. Fix these arguments and call the tool again."}
        result = fn(**args.model_dump(mode="json", exclude_none=True))
        if "error" in result:  # an expected error (unknown SKU, no access): already data, pass it on
            return result
        try:
            result_model.model_validate(result)
        except ValidationError as e:
            print(f"   [BUG] {getattr(fn, '__name__', 'tool')} returned a bad result: {problems(e)}")
            return {"error": "internal_error",
                    "message": "The tool failed. It is not caused by your arguments: don't retry the same call."}
        return result
    return call


SPECS = {  # name -> (args model, result model). ask_user stays as it is: it raises, it returns nothing.
    "get_stock": (StockArgs, StockResult),
    "list_movements": (MovementArgs, MovementsResult),
    "get_invoice": (InvoiceArgs, InvoiceResult),
    "search_products": (SearchArgs, SearchResult),
    "list_low_stock": (LowStockArgs, LowStockResult),
    "resolve_period": (DateSpec, PeriodResult),
    "list_unpaid_invoices": (UnpaidArgs, UnpaidResult),  # 7.2
    "list_suppliers": (SupplierArgs, SupplierResult),  # 7.3
}

NEW_TOOLS = {  # tools added after V6: name -> (function, description)
    "list_unpaid_invoices": (list_unpaid_invoices, UNPAID_DESCRIPTION),
    "list_suppliers": (list_suppliers, SUPPLIERS_DESCRIPTION),
}
RAW_TOOLS = {**TOOLS_V6, **{n: safe(fn) for n, (fn, _) in NEW_TOOLS.items()}}
DESCRIPTIONS = {**{n: d.description for n, d in DECLS_V6.items()}, **{n: d for n, (_, d) in NEW_TOOLS.items()}}

TOOLS_V7 = {**RAW_TOOLS, **{n: typed(RAW_TOOLS[n], a, r) for n, (a, r) in SPECS.items()}}
DECLS_V7 = {**DECLS_V6, **{n: types.FunctionDeclaration(  # same descriptions, schema FROM the Args model
    name=n, description=DESCRIPTIONS[n], parameters_json_schema=a.model_json_schema(),
) for n, (a, _) in SPECS.items()}}
CONFIG_V7 = CONFIG_V6.model_copy(update={"tools": [types.Tool(function_declarations=list(DECLS_V7.values()))]})


# --- Experiments -------------------------------------------------------------------------------

BAD_CALLS = [
    ("list_movements", {"sku": "HAR-0018", "warehouse_code": "BLR", "limit": -5}),
    ("list_movements", {"sku": "HAR-0018", "warehouse_code": "BLR", "date_from": "2026-02-31"}),
    ("list_movements", {"sku": "HAR-0018", "warehouse_code": "BLR", "date_from": "2026-11-01", "date_to": "2026-10-01"}),
    ("get_stock", {"sku": "HAR-0018", "warehouse_code": "Bangalore"}),
    ("get_stock", {"sku": "padlock", "warehouse_code": "BLR"}),
    ("get_stock", {"sku": " har-0018 ", "warehouse_code": "blr"}),  # harmless: fixed, not rejected
    ("get_invoice", {"number": "558"}),
    ("get_stock", {"sku": "HAR-0018", "warehouse_code": "BLR", "product": "padlock"}),
]


def short(result: dict) -> str:
    if "problems" in result:
        return "; ".join(f"{p['field']}: {p['problem']} (got {p['got']!r})" for p in result["problems"])
    if "error" in result:
        return f"{result['error']}: {str(result.get('message', result.get('hint', '')))[:90]}"
    rows = f", rows {result['rows_shown']}" if "rows_shown" in result else ""
    return f"OK{rows}" + (f", quantity {result['quantity']}" if "quantity" in result else "")


QUESTIONS = [
    "Show me the last 200 padlock movements in Bangalore, every row.",
    "Padlock movements in Bangalore from 31 February 2026 to 15 March 2026.",
]


def main():
    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: the same bad arguments, old tools vs typed tools")
    for name, args in BAD_CALLS:
        print(f"\n   {name}({args})")
        print(f"     old:   {short(TOOLS_V6[name](**args))}")
        print(f"     typed: {short(TOOLS_V7[name](**args))}")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: a buggy tool (returns quantity 'many')")
    buggy = typed(lambda **a: {"sku": a["sku"], "name": "Padlock", "warehouse": "BLR", "quantity": "many"},
                  StockArgs, StockResult)
    print(f"   model receives: {buggy(sku='HAR-0018', warehouse_code='BLR')}")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: the agent corrects itself")
    for q in QUESTIONS:
        log: list = []
        reply = as_reply(ask(q, config=CONFIG_V7, registry=recording(TOOLS_V7, log)))
        print(f"\n   calls for {q!r}:")
        for name, args, result in log:
            print(f"     {'XX' if 'error' in result else 'ok'} {name}({dict(args)})\n        -> {short(result)}")
        print(f"   A: {reply[:300]}")


if __name__ == "__main__":
    main()
