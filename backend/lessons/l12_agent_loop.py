"""
Lesson 1.2: The agent loop
===========================

Run from backend/:   uv run python -m lessons.l12_agent_loop

In 1.1 the model ASKED us to run get_stock, and we just printed the request.
Now we close the loop:

    model -> function_call -> OUR code runs SQL -> result back to model -> model again
          -> ... until the model replies with plain text (the final answer)

That loop is all an "agent" is. MAX_STEPS stops it from running forever.

Experiments:
  1. One question, step by step, and checked against the DB
  2. A question that needs TWO tool calls
  3. The MAX_STEPS guard firing
"""

import json
import time

from app.db import query
from app.llm import MODEL, client, types
from lessons.common import header
from lessons.l11_function_call import GET_STOCK_V2  # the declaration from 1.1: what the model reads

MAX_STEPS = 5  # guard: an agent that never stops burns money and time


# --- The real tool: plain Python + SQL. The model never sees this code. -------

def get_stock(sku: str, warehouse_code: str) -> dict:
    rows = query(
        """
        SELECT p.sku, p.name, w.code AS warehouse,
               COALESCE(s.quantity, 0) AS quantity  -- no stock_levels row = not stocked there = 0
        FROM products p
        JOIN warehouses w ON w.code = :wh
        LEFT JOIN stock_levels s ON s.product_id = p.id AND s.warehouse_id = w.id
        WHERE p.sku = :sku
        """,
        sku=sku, wh=warehouse_code,
    )
    if not rows:
        # Return the problem AS DATA so the model can read it (more on this in 2.2)
        return {"error": f"unknown sku {sku!r} or warehouse {warehouse_code!r}"}
    return rows[0]


# Registry: tool name (what the model says) -> Python function (what we run).
# Only names in here can ever run. This is your allow-list.
TOOL_REGISTRY = {"get_stock": get_stock}

CONFIG = types.GenerateContentConfig(
    system_instruction="You are a warehouse assistant. Use tools for any stock numbers; never guess them.",
    tools=[types.Tool(function_declarations=[GET_STOCK_V2])],
)


class TooManySteps(Exception):
    pass


# --- THE AGENT LOOP -----------------------------------------------------------

def run_agent(
    question: str,
    config: types.GenerateContentConfig = CONFIG,
    registry: dict = TOOL_REGISTRY,
    max_steps: int = MAX_STEPS,
    tracer=None,  # optional: something with model_call / tool_call / finish (lesson 1.4)
) -> str:
    # The conversation we build up. Remember 0.2: the model is stateless,
    # so every call resends EVERYTHING, including earlier tool calls and results.
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=question)])
    ]
    print(f"\nQ: {question}")

    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)
        if tracer:
            tracer.model_call(step, response, time.perf_counter() - t0)
        tokens = response.usage_metadata.prompt_token_count
        print(f"   [step {step}] model call #{step}  (prompt tokens: {tokens})")

        # 1. No function call -> the model is done. Its text is the answer.
        if not response.function_calls:
            print(f"   [step {step}] model -> FINAL ANSWER")
            answer = response.text.strip()
            if tracer:
                tracer.finish(answer=answer)
            return answer

        # 2. Keep the model's turn (with its function_call) in the history.
        #    Without this, the next call wouldn't know what it asked for.
        contents.append(response.candidates[0].content)

        # 3. Run every requested call with OUR code, and collect the results.
        result_parts = []
        for call in response.function_calls:
            print(f"   [step {step}] model -> wants {call.name}({call.args})")
            fn = registry.get(call.name)
            t0 = time.perf_counter()
            # pyrefly: ignore[bad-unpacking]
            result = fn(**call.args) if fn else {"error": f"no such tool {call.name!r}"}
            if tracer:
                tracer.tool_call(step, call.name, call.args, result, time.perf_counter() - t0)
            preview = json.dumps(result)
            print(f"   [step {step}] code  -> ran it, result: {preview[:200]}{'...' if len(preview) > 200 else ''}")
            result_parts.append(types.Part.from_function_response(name=call.name, response=result))

        # 4. Feed the evidence back. Then loop: the model sees it on the next call.
        contents.append(types.Content(role="user", parts=result_parts))

    if tracer:
        tracer.finish(error=f"no final answer after {max_steps} steps")
    raise TooManySteps(f"no final answer after {max_steps} steps")


def db_truth(sku: str, wh: str) -> int:
    """Ask the DB directly, with no AI involved, to check the agent's answer."""
    return get_stock(sku, wh)["quantity"]


def main():
    # --- Experiment 1 -------------------------------------------------------------
    header("Experiment 1: one question, full loop, checked against the DB")
    answer = run_agent("How many padlocks are in Bengaluru?")
    print(f"\n   AGENT : {answer}")
    print(f"   DB    : {db_truth('HAR-0018', 'BLR')}  <- the number must match")

    # --- Experiment 2 -------------------------------------------------------------
    header("Experiment 2: a question that needs more than one tool call")
    answer = run_agent("Compare padlock stock in Bengaluru, Mumbai and Delhi. Which needs restocking?")
    print(f"\n   AGENT : {answer}")
    print(f"   DB    : BLR={db_truth('HAR-0018', 'BLR')}  MUM={db_truth('HAR-0018', 'MUM')}"
          f"  DEL={db_truth('HAR-0018', 'DEL')}")
    print("\n-> Did the model ask for all 3 in ONE step (parallel calls) or one per step?")
    print("   Either way, count the model calls: tool steps + 1 final.")

    # --- Experiment 3 -------------------------------------------------------------
    header("Experiment 3: the MAX_STEPS guard")
    try:
        run_agent("How many padlocks are in Delhi?", max_steps=1)
    except TooManySteps as e:
        print(f"\n   STOPPED: {e}")
        print("   -> 1 step is only enough to REQUEST the tool, not to answer with its result.")
        print("      In production this guard stops loops that never end (and runaway bills).")


if __name__ == "__main__":
    main()
