"""
Lesson 3.3: Clarifying questions
=================================

Run from backend/:   uv run python -m lessons.l33_clarify

"stock of box" matches 3 products equally. In 3.1 the model sometimes asked,
because the tool description told it to. Sometimes isn't good enough: a wrong
guess gives a confident, wrong number.

Now CODE decides when to ask:
  - search_products returns status "match" (one clear winner) or "ambiguous"
    (several products score about the same) with the options.
  - the model can't answer from an ambiguous result. It must call ask_user,
    which STOPS the loop and returns a Clarification {question, options} to
    the app, instead of a text answer.

Experiments:
  1. Code decides: which searches are clear, which are ambiguous? (no AI)
  2. The agent asks instead of guessing: "stock of box in Delhi"
  3. The user picks an option, and the conversation continues
  4. A clear name ("padlok") gets answered directly, with no needless question
"""

from pydantic import BaseModel

from app.llm import types
from lessons.common import header
from lessons.l12_agent_loop import run_agent
from lessons.l13_tools import MAX_STEPS
from lessons.l31_fuzzy_search import DECLS_V3, safe, search_products_fuzzy
from lessons.l32_dates import CONFIG_V4, RESOLVE_DECL, TOOLS_V4

MARGIN = 0.05  # scores within 0.05 of the top one count as "the same"


# --- 1. Code decides if a search is ambiguous ---------------------------------------

def search_products_v5(query_text: str) -> dict:
    found = search_products_fuzzy(query_text)  # 3.1's fuzzy search
    matches = found["matches"]
    if not matches:
        return {"status": "not_found", **found}
    top = matches[0]["score"]
    tied = [m for m in matches if top - m["score"] <= MARGIN]
    if len(tied) == 1:
        return {"status": "match", "product": tied[0]}
    return {
        "status": "ambiguous",
        "options": [{"sku": m["sku"], "name": m["name"]} for m in tied],
        "instruction": "Do NOT pick one. Call ask_user with these options.",
    }


# --- 2. ask_user: the model's way to hand a question back to the user --------------

class Clarification(BaseModel):
    """What the app gets instead of a text answer. A UI can show options as buttons (9.2)."""
    question: str
    options: list[dict]

    def as_text(self) -> str:
        lines = [self.question] + [f"{i}. {o['name']} ({o['sku']})" for i, o in enumerate(self.options, 1)]
        return "\n".join(lines)


class NeedsClarification(Exception):
    def __init__(self, clarification: Clarification):
        self.clarification = clarification


def ask_user(question: str, options: list[dict]) -> dict:
    # Raising stops 1.2's loop right here: no more model calls, no guessing.
    # (Not wrapped in safe(), which would turn this exception back into data.)
    raise NeedsClarification(Clarification(question=question, options=options))


SEARCH_DECL = types.FunctionDeclaration(
    name="search_products",
    description=(
        "Find products by name, even with typos. Use this FIRST whenever the user names a product. "
        "Returns status 'match' (use product.sku), 'ambiguous' (several equally good matches: you "
        "MUST call ask_user with the options, never pick one yourself) or 'not_found'."
    ),
    parameters_json_schema=DECLS_V3["search_products"].parameters_json_schema,
)
ASK_USER_DECL = types.FunctionDeclaration(
    name="ask_user",
    description=(
        "Ask the user to choose when their request is ambiguous. Pass a short question and the "
        "options exactly as the tool returned them. This ends your turn; the user answers next."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "e.g. 'Which box do you mean?'"},
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}, "name": {"type": "string"}},
                    "required": ["sku", "name"],
                },
            },
        },
        "required": ["question", "options"],
    },
)

TOOLS_V5 = {**TOOLS_V4, "search_products": safe(search_products_v5), "ask_user": ask_user}
DECLS_V5 = {**DECLS_V3, "resolve_period": RESOLVE_DECL, "search_products": SEARCH_DECL, "ask_user": ASK_USER_DECL}
CONFIG_V5 = CONFIG_V4.model_copy(update={"tools": [types.Tool(function_declarations=list(DECLS_V5.values()))]})


def ask(question: str, tracer=None, history=None, config=CONFIG_V5, registry=TOOLS_V5,
        prepare=None) -> str | Clarification:
    """A text answer, or a Clarification when the agent needs the user to choose."""
    try:
        return run_agent(question, config=config, registry=registry, max_steps=MAX_STEPS,
                         tracer=tracer, history=history, prepare=prepare)
    except NeedsClarification as e:
        if tracer:
            tracer.finish(answer=e.clarification.as_text())
        return e.clarification


def turn(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def main():
    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: code decides clear vs ambiguous (no AI)")
    for q in ["padlok", "box", "corugated box", "brass", "ball valv", "hammer"]:
        r = search_products_v5(q)
        detail = (r["product"]["name"] if r["status"] == "match"
                  else ", ".join(o["name"] for o in r["options"]) if r["status"] == "ambiguous" else "")
        print(f"   {q:<15}{r['status']:<11}{detail}")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: ambiguous -> the agent asks instead of guessing")
    question = "What's the stock of corugated box in Delhi?"
    result = ask(question)
    print(f"\n   returned: {type(result).__name__}")
    print("   " + (result.as_text() if isinstance(result, Clarification) else result).replace("\n", "\n   "))

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: the user picks, the conversation continues")
    if isinstance(result, Clarification):
        history = [turn("user", question), turn("model", result.as_text())]
        print(ask("the 12x12x12 one", history=history))

    # --- Experiment 4 ---------------------------------------------------------
    header("Experiment 4: a clear name gets answered directly")
    print(ask("how many padlok in bangalore?"))


if __name__ == "__main__":
    main()
