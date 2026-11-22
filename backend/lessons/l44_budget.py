"""
Lesson 4.4: Context assembly and token budget
==============================================

Run from backend/:   uv run python -m lessons.l44_budget

Every model call is built from the same parts, always in the same order:

    system   base instructions                    fixed
    tools    tool declarations                    fixed
    memory   4.3 summary + 4.2 selection/pending  small, grows slowly
    history  recent chat messages                 grows per turn (4.3 compacts it)
    question the new message                      small
    work     THIS turn's tool calls + results     grows per STEP, fastest of all

build_context() (run by 1.2's loop before EVERY model call, via its `prepare` hook)
counts tokens per part, prints the breakdown, and trims a part that is over budget:
    - one tool result too big     -> keep the first rows that fit, say how many were cut
    - all tool work too big       -> drop results of OLDER steps (the model can call again)
    - history too big             -> drop the oldest messages (before the turn starts)
system, tools, question and the selection are never trimmed.

Experiments:
  1. A normal investigation: watch the parts grow step by step
  2. One huge tool result (50 movements) gets trimmed to fit
  3. A long investigation (3 warehouses x 50 movements): older results get dropped
"""

import json
from functools import lru_cache

from app.llm import MODEL, client, types
from lessons.common import header
from lessons.l41_history import new_session
from lessons.l42_selections import CONFIG_V6
from lessons.l42_selections import chat as chat_with_state
from lessons.l43_compaction import compacted_history, ensure_tables, summary_prompt

BUDGET = {"memory": 400, "history": 1200, "work": 1500}  # tokens; system/tools/question are never trimmed
RESULT_MAX = 800  # tokens for ONE tool result


@lru_cache(maxsize=4096)
def tokens(text: str) -> int:
    """Real token count from Gemini (cached: the same text is never counted twice)."""
    return client.models.count_tokens(model=MODEL, contents=text).total_tokens if text else 0


def part_text(p: types.Part) -> str:
    if p.text:
        return p.text
    if p.function_call:
        return json.dumps({"call": p.function_call.name, "args": p.function_call.args})
    if p.function_response:
        return json.dumps(p.function_response.response)
    return ""


def count(contents: list[types.Content]) -> int:
    return sum(tokens(part_text(p)) for c in contents for p in (c.parts or []))


# --- Trimming ---------------------------------------------------------------------------

def shrink(result: dict, max_tokens: int) -> dict:
    """Too big? Keep the first rows of its longest list that fit, and say what was cut."""
    if tokens(json.dumps(result)) <= max_tokens:
        return result
    lists = [k for k, v in result.items() if isinstance(v, list)]
    if not lists:
        return {"trimmed": "result too large to show", "preview": json.dumps(result)[: max_tokens * 3]}
    key = max(lists, key=lambda k: len(result[k]))
    rows, keep = result[key], len(result[key])
    while keep > 1 and tokens(json.dumps({**result, key: rows[:keep]})) > max_tokens:
        keep //= 2
    return {**result, key: rows[:keep],
            "trimmed": f"showing {keep} of {len(rows)} {key}. Ask with a narrower filter (dates) for the rest."}


def with_response(p: types.Part, response: dict) -> types.Part:
    return types.Part.from_function_response(name=p.function_response.name, response=response)


def trim_work(work: list[types.Content]) -> tuple[list[types.Content], list[str]]:
    """This turn's tool calls and results -> within BUDGET['work']. Returns (new work, what was trimmed)."""
    notes = []
    # 1. every single result <= RESULT_MAX
    out = []
    for c in work:
        parts = []
        for p in c.parts or []:
            if p.function_response and "trimmed" not in p.function_response.response:
                small = shrink(p.function_response.response, RESULT_MAX)
                if small is not p.function_response.response:
                    notes.append(f"{p.function_response.name}: {small['trimmed']}")
                    p = with_response(p, small)
            parts.append(p)
        out.append(types.Content(role=c.role, parts=parts))
    # 2. still over? drop results of older steps, oldest first (the latest step is what the model needs now)
    last_results = max((i for i, c in enumerate(out) if any(p.function_response for p in c.parts)), default=-1)
    for i, c in enumerate(out):
        if count(out) <= BUDGET["work"] or i >= last_results:
            break
        for j, p in enumerate(c.parts):
            if p.function_response and p.function_response.response.get("trimmed") != "dropped":
                c.parts[j] = with_response(p, {"trimmed": "dropped",
                                               "hint": "older result removed to save space; call the tool again if needed"})
                notes.append(f"{p.function_response.name}: older result dropped")
    return out, notes


def trim_history(history: list[types.Content]) -> list[types.Content]:
    while len(history) > 2 and count(history) > BUDGET["history"]:
        history = history[2:]  # drop the oldest user+assistant pair
    return history


def trim_text(text: str, max_tokens: int) -> str:
    while tokens(text) > max_tokens:
        text = text[: int(len(text) * 0.8)]
    return text


# --- build_context: runs before EVERY model call -------------------------------------------

def tools_tokens(config) -> int:
    decls = [d.model_dump(exclude_none=True) for d in config.tools[0].function_declarations]
    return tokens(json.dumps(decls))  # cached by tokens(): the same tool set is counted once


def build_context(n_history: int, on_context=None):
    """Returns the `prepare` hook for 1.2's loop. contents = history | question | this turn's work.
    on_context(contents, config, parts): 4.8 sees the final context of every call, to test it."""
    base = CONFIG_V6.system_instruction

    def prepare(contents: list[types.Content], config) -> list[types.Content]:
        history, question, work = contents[:n_history], contents[n_history], contents[n_history + 1:]
        work, notes = trim_work(work)
        parts = {
            "system": tokens(base),
            "tools": tools_tokens(config),
            "memory": tokens(config.system_instruction[len(base):]),
            "history": count(history),
            "question": count([question]),
            "work": count(work),
        }
        line = " | ".join(f"{k} {v}" for k, v in parts.items())
        print(f"   [context] {line} | total ~{sum(parts.values())}")
        for n in notes:
            print(f"   [trimmed] {n}")
        contents = [*history, question, *work]
        if on_context:
            on_context(contents, config, parts)
        return contents

    return prepare


def chat(session_id: int, text: str, tracer=None, log: list | None = None,
         tools: list[str] | None = None, use_history: bool = True,
         extra_prompt: str = "", extra_tools: dict | None = None, on_context=None, **extra) -> str:
    """4.3's turn (compacted history + summary) + 4.2's state, with every part on a budget.
    tools / use_history: 4.5 sends only what the question needs (default: everything).
    use_history=False still keeps the last user+assistant pair, but drops older turns and the summary.
    extra_prompt / extra_tools: 4.6's long-term memory and its tools, passed through to 4.2.
    **extra (4.7's scope) goes straight to 4.2."""
    history, summary, _ = compacted_history(session_id)
    history = trim_history(history) if use_history else history[-2:]  # always keep the last turn: cheap, and it holds the SKU
    memory = trim_text(summary_prompt(summary), BUDGET["memory"]) if use_history else ""
    return chat_with_state(session_id, text, tracer=tracer, log=log, history=history,
                           extra_prompt=extra_prompt + memory, prepare=build_context(len(history), on_context),
                           tools=tools, extra_tools=extra_tools, **extra)


def main():
    ensure_tables()
    print(f"   budgets: {BUDGET}, one tool result <= {RESULT_MAX}")

    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: a normal investigation, part by part")
    sid = new_session()
    print(chat(sid, "How many padlocks do we have in Bangalore?"))
    print(chat(sid, "Why is it dropping?"))
    print("\n-> [context] = our count per part. 'prompt tokens' = Gemini's real total for the same call.")
    print("   'work' is the part that grows inside a turn: every step adds a call + its result.")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: one huge tool result")
    print(chat(new_session(), "List the last 50 padlock movements in Bangalore, every row."))

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: a long investigation (3 warehouses)")
    print(chat(new_session(), "Get the last 50 padlock movements in Bangalore, then Mumbai, then Delhi, "
                              "one warehouse at a time, and compare them."))


if __name__ == "__main__":
    main()
