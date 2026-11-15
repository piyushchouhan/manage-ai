"""
Lesson 3.2: Dates and relative time
====================================

Run from backend/:   uv run python -m lessons.l32_dates

"last month", "this quarter", "since 1 Oct": tools need exact ISO dates.

Split the job:
  - the MODEL turns words into a small structure (what it's good at):
        "last quarter" -> {kind: "period", unit: "quarter", offset: -1}
  - CODE turns the structure into dates (calendar maths, always right):
        -> 2026-07-01 .. 2026-09-30

Experiments:
  1. 12 date phrases: model calculates dates itself vs model-structure + code-maths
  2. The agent gets a resolve_period tool: "how many padlocks did we sell in BLR last month?"
"""

import calendar
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from app.db import query
from app.llm import MODEL, client, types
from lessons.common import header
from lessons.l12_agent_loop import run_agent
from lessons.l13_tools import MAX_STEPS
from lessons.l31_fuzzy_search import CONFIG_V3, DECLS_V3, TOOLS_V3, safe

TEST_TODAY = datetime.strptime("15/11/2026", "%d/%m/%Y").date()  # fixed, so the test gives the same answers on any day (a Sunday)


# --- The structure the model fills in (no arithmetic needed) -------------------

class DateSpec(BaseModel):
    kind: Literal["period", "last_n", "since", "between", "none"] = Field(description=(
        "period = a calendar unit like 'last month', 'this quarter', 'yesterday', 'in March'; "
        "last_n = a rolling window like 'last 7 days', 'past 3 months'; "
        "since = from a date until today; between = two explicit dates; none = no date mentioned."
    ))
    unit: Literal["day", "week", "month", "quarter", "year"] | None = Field(None, description="For period and last_n.")
    offset: int | None = Field(None, description="For period: 0 = this, -1 = last/previous, -2 = the one before. "
                                                 "'yesterday' = unit day, offset -1.")
    n: int | None = Field(None, description="For last_n: how many units.")
    month: int | None = Field(None, description="For a named month without a date ('in March'): 1-12. Use with kind=period, unit=month.")
    start: date | None = Field(None, description="For since/between: the date as written. If no year is given, use the current year.")
    end: date | None = Field(None, description="For between: the end date as written.")


# --- The maths, in code -----------------------------------------------------------

def shift_months(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    return date(y, m + 1, 1)


def resolve(spec: DateSpec, today: date) -> tuple[date, date] | None:
    """DateSpec -> (date_from, date_to), both inclusive. Never later than today."""
    if spec.kind == "none":
        return None
    if spec.kind == "since":
        start = spec.start.replace(year=today.year)
        if start > today:  # "since 20 Dec" said in November means LAST December
            start = start.replace(year=today.year - 1)
        return start, today
    if spec.kind == "between":
        return spec.start, spec.end
    if spec.kind == "last_n":  # rolling window that includes today: "last 7 days" = today and the 6 before
        if spec.unit == "day":
            return today - timedelta(days=spec.n - 1), today
        if spec.unit == "week":
            return today - timedelta(weeks=spec.n) + timedelta(days=1), today
        months = {"month": 1, "quarter": 3, "year": 12}[spec.unit] * spec.n
        return shift_months(today, -months).replace(day=min(today.day, 28)) + timedelta(days=1), today

    # kind == "period": a whole calendar unit
    if spec.month:  # "in March": the most recent March that has started
        year = today.year if spec.month <= today.month else today.year - 1
        start = date(year, spec.month, 1)
        end = date(year, spec.month, calendar.monthrange(year, spec.month)[1])
    else:
        off = spec.offset or 0
        if spec.unit == "day":
            start = end = today + timedelta(days=off)
        elif spec.unit == "week":  # weeks start on Monday
            start = today - timedelta(days=today.weekday()) + timedelta(weeks=off)
            end = start + timedelta(days=6)
        elif spec.unit in ("month", "quarter"):
            size = 1 if spec.unit == "month" else 3
            first_of_unit = date(today.year, (today.month - 1) // size * size + 1, 1)
            start = shift_months(first_of_unit, off * size)
            end = shift_months(start, size) - timedelta(days=1)
        else:  # year
            start, end = date(today.year + off, 1, 1), date(today.year + off, 12, 31)
    return start, min(end, today)  # "this month" stops at today, not the 30th


# --- Two ways to get dates from a phrase ------------------------------------------

def extract_spec(phrase: str, today: date) -> DateSpec:
    """Model: words -> DateSpec (structured output, lesson 0.3)."""
    r = client.models.generate_content(
        model=MODEL, contents=phrase,
        config=types.GenerateContentConfig(
            system_instruction=f"Today is {today.isoformat()} ({today:%A}). Describe the time period in the text. Don't calculate dates.",
            response_mime_type="application/json", response_schema=DateSpec,
        ),
    )
    return DateSpec.model_validate_json(r.text)


class ModelDates(BaseModel):
    date_from: date | None
    date_to: date | None


def model_does_maths(phrase: str, today: date) -> tuple[date, date] | None:
    """Model: words -> final dates, calculated 'in its head'."""
    r = client.models.generate_content(
        model=MODEL, contents=phrase,
        config=types.GenerateContentConfig(
            system_instruction=(f"Today is {today.isoformat()} ({today:%A}). Give the inclusive date range the text "
                                "refers to. Weeks start Monday. Ranges never go past today. Nulls if no date."),
            response_mime_type="application/json", response_schema=ModelDates,
        ),
    )
    d = ModelDates.model_validate_json(r.text)
    return (d.date_from, d.date_to) if d.date_from else None


def D(dmy_text: str) -> date:
    """D("01/10/2026") -> date(2026, 10, 1). Lets tests be written as DD/MM/YYYY."""
    return datetime.strptime(dmy_text, "%d/%m/%Y").date()


TESTS = [  # phrase -> expected (date_from, date_to) for TEST_TODAY = Sunday 15/11/2026
    ("sales last month",                     (D("01/10/2026"), D("31/10/2026"))),
    ("movements this month",                 (D("01/11/2026"), D("15/11/2026"))),
    ("this quarter",                         (D("01/10/2026"), D("15/11/2026"))),
    ("last quarter",                         (D("01/07/2026"), D("30/09/2026"))),
    ("since 1 Oct",                          (D("01/10/2026"), D("15/11/2026"))),
    ("in the last 7 days",                   (D("09/11/2026"), D("15/11/2026"))),
    ("yesterday",                            (D("14/11/2026"), D("14/11/2026"))),
    ("last week",                            (D("02/11/2026"), D("08/11/2026"))),
    ("this year so far",                     (D("01/01/2026"), D("15/11/2026"))),
    ("between 5 and 20 September",           (D("05/09/2026"), D("20/09/2026"))),
    ("what happened in March",               (D("01/03/2026"), D("31/03/2026"))),
    ("how many padlocks do we have",         None),
]


# --- The agent: resolve_period as a tool. The model fills DateSpec as tool args. ---

def resolve_period(**spec) -> dict:
    today = date.today()
    rng = resolve(DateSpec(**spec), today)
    if rng is None:
        return {"date_from": None, "date_to": None}
    return {"date_from": rng[0].isoformat(), "date_to": rng[1].isoformat()}


RESOLVE_DECL = types.FunctionDeclaration(
    name="resolve_period",
    description=(
        "Turn a time phrase from the user ('last month', 'this quarter', 'since 1 Oct') into exact "
        "dates. ALWAYS use this instead of calculating dates yourself, then pass date_from/date_to "
        "to other tools. Describe the phrase; don't calculate."
    ),
    parameters_json_schema=DateSpec.model_json_schema(),  # same fields as the extraction step
)
TOOLS_V4 = {**TOOLS_V3, "resolve_period": safe(resolve_period)}
CONFIG_V4 = CONFIG_V3.model_copy(update={
    "tools": [types.Tool(function_declarations=[*DECLS_V3.values(), RESOLVE_DECL])]
})


def ask(question: str, tracer=None, history=None) -> str:
    return run_agent(question, config=CONFIG_V4, registry=TOOLS_V4, max_steps=MAX_STEPS,
                     tracer=tracer, history=history)


def dmy(d: date) -> str:
    return d.strftime("%d/%m/%Y")  # display format; tools and SQL keep ISO


def fmt(rng) -> str:
    return f"{dmy(rng[0])} .. {dmy(rng[1])}" if rng else "none"


def main():
    # --- Experiment 1 ---------------------------------------------------------
    header(f"Experiment 1: 12 phrases, today = {dmy(TEST_TODAY)} ({TEST_TODAY:%A})")
    print(f"   {'phrase':<30}{'expected':<26}{'model does maths':<28}{'model structure + code'}")
    score = {"model": 0, "code": 0}
    for phrase, expected in TESTS:
        by_model = model_does_maths(phrase, TEST_TODAY)
        by_code = resolve(extract_spec(phrase, TEST_TODAY), TEST_TODAY)
        score["model"] += by_model == expected
        score["code"] += by_code == expected
        mark = lambda got: ("ok " if got == expected else "XX ") + fmt(got)
        print(f"   {phrase:<30}{fmt(expected):<26}{mark(by_model):<28}{mark(by_code)}")
    print(f"\n   correct:  model does maths {score['model']}/{len(TESTS)}   "
          f"structure + code {score['code']}/{len(TESTS)}")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: the agent uses resolve_period, then list_movements")
    print(ask("How many padlocks did we sell in BLR last month?"))
    first, last = resolve(DateSpec(kind="period", unit="month", offset=-1), date.today())
    truth = query(
        """SELECT COALESCE(-SUM(m.quantity), 0) AS sold FROM stock_movements m
           JOIN products p ON p.id = m.product_id JOIN warehouses w ON w.id = m.warehouse_id
           WHERE p.sku = 'HAR-0018' AND w.code = 'BLR' AND m.movement_type = 'sale_out'
             AND m.occurred_at::date BETWEEN :a AND :b""",
        a=first, b=last,
    )[0]["sold"]
    print(f"\n   DB truth: {truth} padlocks sold in BLR between {dmy(first)} and {dmy(last)}")


if __name__ == "__main__":
    main()
