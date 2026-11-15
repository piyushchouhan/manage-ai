"""
Lesson 1.1: Function calling (no execution yet)
================================================

Run from backend/:   uv run python -m lessons.l11_function_call

The model can't see your database. A TOOL is a function you DESCRIBE to the
model (name + description + JSON-schema parameters). When the model thinks it
needs that data, it doesn't answer. It replies with a function_call:
"please run get_stock with these args". It never runs anything itself.

This is the same trick as 0.3 (structured output). The tool's parameter
schema is the JSON shape, and the model fills it in.

Experiments:
  1. Declare get_stock -> ask "how many padlocks in BLR?" -> print the call
  2. Same question, better descriptions -> different arguments
  3. The model decides WHETHER to call: tool vs plain text vs forced call
"""

from app.llm import MODEL, client, types
from lessons.common import header


# --- The tool declaration: this is ALL the model knows about your function ----
# No Python code is sent. Only this description. The model reads it the way
# you'd read API docs.

GET_STOCK_V1 = types.FunctionDeclaration(
    name="get_stock",
    description="Get the current stock quantity of a product in a warehouse.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "sku": {"type": "string", "description": "Product SKU."},
            "warehouse_code": {"type": "string", "description": "Warehouse code."},
        },
        "required": ["sku", "warehouse_code"],
    },
)


def ask(question: str, tool: types.FunctionDeclaration, mode: str = "AUTO"):
    """Send ONE question with ONE tool available. Print what comes back. Execute nothing."""
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[tool])],
            # AUTO = model decides; ANY = must call a tool; NONE = may not call tools
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=mode)
            ),
        ),
    )
    print(f"\nQ: {question}   (mode={mode})")
    if response.function_calls:
        for call in response.function_calls:
            # call.args is a plain Python dict: ready to become **kwargs in 1.2
            print(f"   -> FUNCTION CALL: {call.name}({call.args})")
        print("   (nothing ran. The model is only ASKING us to run it.)")
    else:
        print(f"   -> TEXT: {(response.text or '').strip()[:200]}")
    return response

# Same function, same code behind it later. Only the TEXT the model reads changes.
GET_STOCK_V2 = types.FunctionDeclaration(
    name="get_stock",
    description=(
        "Get the current on-hand stock quantity of ONE product in ONE warehouse. "
        "Use this for questions like 'how many X do we have in Y'."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "sku": {
                "type": "string",
                "description": (
                    "Exact product SKU, format AAA-0000. Known SKUs: "
                    "HAR-0018 = Padlock 50mm Brass."
                ),
            },
            "warehouse_code": {
                "type": "string",
                "enum": ["BLR", "MUM", "DEL"],  # the model can ONLY pick one of these
                "description": "BLR = Bengaluru/Bangalore, MUM = Mumbai/Bombay, DEL = Delhi.",
            },
        },
        "required": ["sku", "warehouse_code"],
    },
)


def main():
    # --- Experiment 1 -------------------------------------------------------------
    header("Experiment 1: the model returns a function call, not an answer")
    r = ask("how many padlocks in BLR?", GET_STOCK_V1)

    # Models are not deterministic. With a vague tool description, the SAME question
    # sometimes gets a guessed call and sometimes a question back. Never assume a call came back.
    if r.function_calls:
        # Peek under the hood: a function call is just another kind of "part" in the reply
        part = next(p for p in r.candidates[0].content.parts if p.function_call)
        print(f"\n   raw part : {part.function_call.model_dump(exclude_none=True)}")
        print("\n-> Look at the sku. The model has never seen our product table, so")
        print("   whatever is there is a GUESS. (The real SKU is HAR-0018.)")
    else:
        print("\n-> No call this time: the model knew it didn't have a real SKU, so it asked.")
        print("   Run again and it may GUESS instead (e.g. sku='padlocks'). Both come from")
        print("   the same cause: the description gives it no way to know the SKU.")


    # --- Experiment 2 -------------------------------------------------------------
    header("Experiment 2: descriptions decide the arguments")


    for q in [
        "how many padlocks in BLR?",
        "do we have brass locks in bangalore",
        "padlock count in bombay",
    ]:
        ask(q, GET_STOCK_V2)
    print("\n-> Same question, better docs, correct args. Hard-coding SKUs in a")
    print("   description doesn't scale, though. Lesson 1.3 adds search_products for that.")


    # --- Experiment 3 -------------------------------------------------------------
    header("Experiment 3: the model decides WHETHER to call a tool")
    ask("hi! what can you help me with?", GET_STOCK_V2)              # no data needed -> text
    ask("how many padlocks in BLR?", GET_STOCK_V2, mode="NONE")      # tools disabled -> must guess/refuse
    ask("hi! what can you help me with?", GET_STOCK_V2, mode="ANY")  # forced -> calls even when silly
    print("\n-> AUTO is what you normally want. ANY/NONE are tools you'll use in routers")
    print("   and orchestration later, when CODE (not the model) decides the next step.")


if __name__ == "__main__":
    main()
