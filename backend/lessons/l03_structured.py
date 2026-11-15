"""
Lesson 0.3: Structured output
================================

Run from backend/:   uv run python -m lessons.l03_structured

Free text is for humans. Code needs DATA. Structured output makes the model
return JSON that matches a schema we define, so the answer can drive code
(routing, SQL filters, tool calls...).

Experiments:
  1. Free text: why it's useless for code
  2. Schema-constrained JSON -> a typed Pydantic object
  3. Pydantic validation catches bad data, even when the JSON looks fine
"""

import json
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.llm import MODEL, client, types
from lessons.common import header

TODAY = date.today()


# --- The schema: this IS the contract between the model and your code --------
# Field descriptions are sent to the model. They are instructions, not comments.

class Intent(str, Enum):
    stock_lookup = "stock_lookup"
    movement_history = "movement_history"
    invoice_lookup = "invoice_lookup"
    payment_status = "payment_status"
    low_stock_report = "low_stock_report"
    other = "other"


class ParsedQuery(BaseModel):
    intent: Intent = Field(description="What the user wants to do.")
    product_text: str | None = Field(
        None, description="The product exactly as the user wrote it (typos included). Null if none."
    )
    customer_name: str | None = Field(None, description="Customer name if mentioned, else null.")
    warehouse_code: Literal["BLR", "MUM", "DEL"] | None = Field(
        None, description="BLR=Bengaluru/Bangalore, MUM=Mumbai/Bombay, DEL=Delhi. Null if not mentioned."
    )
    date_from: date | None = Field(None, description="Start of the date range, ISO format, if any.")
    date_to: date | None = Field(None, description="End of the date range, ISO format, if any.")
    needs_clarification: bool = Field(
        description="True if the request is too vague to act on (e.g. refers to 'that customer' with no context)."
    )
    clarification_question: str | None = Field(
        None, description="The question to ask the user when needs_clarification is true."
    )

    # Business rules the model can't be trusted to follow: enforced by CODE.
    @model_validator(mode="after")
    def check_rules(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from is after date_to")
        if self.date_to and self.date_to > TODAY:
            raise ValueError("date_to is in the future")
        if self.needs_clarification and not self.clarification_question:
            raise ValueError("needs_clarification=true but no question given")
        return self


SYSTEM = (
    f"Today is {TODAY.isoformat()} ({TODAY:%A}). "
    "You parse questions for a warehouse assistant. Extract the fields; do not answer the question."
)


def parse(question: str) -> tuple[ParsedQuery, types.GenerateContentResponse]:
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",  # 1. "reply with JSON only"
            response_schema=ParsedQuery,            # 2. "...shaped exactly like this"
        ),
    )
    # 3. Validate with Pydantic ourselves (runs our model_validator rules too)
    return ParsedQuery.model_validate_json(response.text), response



# --- Experiment 1 -------------------------------------------------------------
header("Experiment 1: free text (what code would have to deal with)")
r = client.models.generate_content(
    model=MODEL,
    contents="Extract the intent, customer, and date range from: "
             "'show me unpaid invoices for Sharma Traders from last month'",
)
print(r.text.strip())
print("\n-> Readable, but how would code get the dates out of THAT? Regex? And next")
print("   time the wording and layout will be different.")

# --- Experiment 2 -------------------------------------------------------------
header("Experiment 2: schema-constrained output -> typed Python object")
QUESTIONS = [
    "show me unpaid invoices for Sharma Traders from last month",
    "how many padlok in bangalore?",
    "stock movements of brass lock in mumbai since 1st oct",
    "what's running low in delhi",
    "how much did that customer pay?",
    "write me a poem about cardboard boxes",
]
for q in QUESTIONS:
    parsed, resp = parse(q)
    print(f"\nQ: {q}")
    print(f"   raw JSON : {resp.text.strip()[:140]}{'...' if len(resp.text) > 140 else ''}")
    fields = parsed.model_dump(exclude_none=True, mode="json")
    print(f"   parsed   : {fields}")
    # This is the payoff: plain attribute access, real types
    if parsed.needs_clarification:
        print(f"   -> code path: ASK USER  -> {parsed.clarification_question!r}")
    else:
        print(f"   -> code path: {parsed.intent.value}"
              f"{f'  (date type: {type(parsed.date_from).__name__})' if parsed.date_from else ''}")

# --- Experiment 3 -------------------------------------------------------------
header("Experiment 3: validation catches bad data before it reaches your code")
bad_outputs = {
    "broken JSON": '{"intent": "stock_lookup", ',
    "unknown intent": '{"intent": "delete_everything", "needs_clarification": false}',
    "unknown warehouse": '{"intent": "stock_lookup", "warehouse_code": "CHN", "needs_clarification": false}',
    "valid JSON, wrong logic": json.dumps({
        "intent": "invoice_lookup", "date_from": "2026-10-31", "date_to": "2026-10-01",
        "needs_clarification": False,
    }),
}
for label, raw in bad_outputs.items():
    try:
        ParsedQuery.model_validate_json(raw)
        print(f"   {label:<24} -> accepted (!)")
    except ValidationError as e:
        print(f"   {label:<24} -> REJECTED: {e.errors()[0]['msg']}")
print("\n-> The schema guarantees SHAPE. Validators guarantee RULES. Neither guarantees")
print("   TRUTH: if the model computes 'last month' wrong, it's still valid. (Lesson 3.2, Module 6)")
