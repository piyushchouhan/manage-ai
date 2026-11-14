"""
Lesson 0.1: Your first Gemini call
====================================

Run from backend/:   uv run python -m lessons.l01_hello

Three experiments, three facts:
  1. A call = messages in -> text + token counts out.
  2. The model is STATELESS. It forgets everything between calls.
  3. A system prompt is just extra instructions sent with every call.
"""

import time

# Shared client + model name (configured once in app/llm.py from the root .env)
from app.llm import MODEL, client, types


def ask(contents, system: str | None = None, temperature: float = 0.0):
    """One round-trip to the model. This function is the whole 'primitive'."""
    config = types.GenerateContentConfig(
        system_instruction=system,  # rules for the model (optional)
        temperature=temperature,    # 0 = most deterministic
    )
    start = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=contents, config=config)
    elapsed = time.perf_counter() - start

    u = response.usage_metadata
    print(f"   answer : {response.text.strip()}")
    print(
        f"   tokens : in={u.prompt_token_count}  out={u.candidates_token_count}"
        f"  thinking={u.thoughts_token_count or 0}  total={u.total_token_count}"
    )
    print(f"   latency: {elapsed:.2f}s")
    return response


def header(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --- Experiment 1: the basic call --------------------------------------------
header("Experiment 1: one question in, one answer out")
q1 = "In one sentence: what is a stock-keeping unit (SKU) in a warehouse?"
print(f"   you    : {q1}")
ask(q1)

# --- Experiment 2: the model has no memory -----------------------------------
header("Experiment 2: a second, separate call")
q2 = "What did I just ask you?"
print(f"   you    : {q2}")
ask(q2)
print("   -> It doesn't know. Every call starts from zero.")
print("      Chat apps fake memory by RESENDING the whole history (Lesson 0.2).")

# --- Experiment 3: system prompt changes behaviour ---------------------------
header("Experiment 3: same question, with a system prompt")
SYSTEM = (
    "You are the assistant for a warehouse company with sites in Bengaluru (BLR), "
    "Mumbai (MUM) and Delhi (DEL). Answer in at most 2 short sentences. "
    "If a question needs live data (stock, invoices, payments), say you don't have "
    "access to the database yet. Never invent numbers."
)
q3 = "How many padlocks do we have in Bengaluru?"
print(f"   you    : {q3}")
print("   [without system prompt]")
ask(q3)
print("   [with system prompt]")
ask(q3, system=SYSTEM)
print("   -> Same model, same question, different behaviour: the system prompt sets the rules.")
print("      Neither version knows the real number. That needs TOOLS (Module 1).")
