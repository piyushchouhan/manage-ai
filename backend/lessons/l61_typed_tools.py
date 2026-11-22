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

from app.llm import types
from lessons.common import header
from lessons.l13_tools import SKU, WAREHOUSE
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
}

TOOLS_V7 = {**TOOLS_V6, **{n: typed(TOOLS_V6[n], a, r) for n, (a, r) in SPECS.items()}}
DECLS_V7 = {**DECLS_V6, **{n: types.FunctionDeclaration(  # same descriptions, schema FROM the Args model
    name=n, description=DECLS_V6[n].description, parameters_json_schema=a.model_json_schema(),
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
