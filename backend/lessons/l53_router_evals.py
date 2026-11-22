"""
Lesson 5.3: Router evals
=========================

Run from backend/:   uv run python -m lessons.l53_router_evals

An EVAL is a test for an AI part. A normal test says "f(2) must be 4". A model can't be
tested like that (it isn't exact and changes when you edit a prompt), so we measure it:

    evals/router.jsonl     36 messages, each with the route a PERSON picked (the "label")
    run the router on all  -> ACCURACY = how many it got right
                           -> CONFUSION MATRIX = which route it confuses with which

The dataset includes what real users send: typos, Hinglish, follow-ups (with the earlier
turn, built exactly like 5.2 builds it), vague messages, and "ok".

Then the loop every AI team runs:  change the prompt -> run the eval -> did the number go up?
Without the number you'd test 2-3 messages by hand and never see what you broke.

Experiments:
  1. The cheap router (5.2's ROUTER_MODEL), prompt v1
  2. The same eval on the main model: is the cheap one good enough?
  3. Prompt v2 (rules written from experiment 1's mistakes, now 5.1's ROUTER_PROMPT): number up? anything broken?
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import get_args

from app.llm import MODEL, ROUTER_MODEL
from lessons.common import header
from lessons.l51_router import ROUTER_PROMPT, ROUTER_PROMPT_V1, Route, classify
from lessons.l52_dispatch import Decision, format_input

EVAL_FILE = Path(__file__).resolve().parents[1] / "evals" / "router.jsonl"
ROUTES = get_args(Route.model_fields["route"].annotation)  # the 4 labels, read from 5.1's schema


def load() -> list[dict]:
    return [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]


def predict(item: dict, model: str, prompt: str) -> str:
    try:
        return classify(format_input(item.get("earlier", []), item["message"]),
                        model=model, schema=Decision, prompt=prompt).route
    except Exception as e:  # a broken output counts as wrong, it doesn't stop the eval
        return f"error: {type(e).__name__}"


def evaluate(model: str, prompt: str = ROUTER_PROMPT) -> list[tuple[dict, str]]:
    items = load()
    with ThreadPoolExecutor(max_workers=8) as pool:  # 36 independent calls: run them in parallel
        got = list(pool.map(lambda it: predict(it, model, prompt), items))
    return list(zip(items, got))


def report(results: list[tuple[dict, str]]) -> float:
    right = sum(item["route"] == got for item, got in results)
    accuracy = right / len(results)
    print(f"\n   accuracy: {right}/{len(results)} = {accuracy:.0%}")

    pairs = Counter((item["route"], got) for item, got in results)
    print("\n   confusion matrix (rows = label, columns = router's answer)")
    print("   " + " " * 15 + "".join(f"{r[:12]:>14}" for r in ROUTES))
    for label in ROUTES:
        print(f"   {label:<15}" + "".join(f"{pairs[(label, got)] or '.':>14}" for got in ROUTES))

    wrong = [(item, got) for item, got in results if item["route"] != got]
    if wrong:
        print("\n   mistakes:")
        for item, got in wrong:
            follow = "  (follow-up)" if item.get("earlier") else ""
            print(f"     {item['message']!r}{follow}\n       label {item['route']}, got {got}")
    return accuracy


def main():
    print(f"   {len(load())} labelled messages from {EVAL_FILE.name}: {dict(Counter(i['route'] for i in load()))}")

    header(f"Experiment 1: cheap router ({ROUTER_MODEL}), prompt v1")
    cheap = report(evaluate(ROUTER_MODEL, ROUTER_PROMPT_V1))

    header(f"Experiment 2: main model ({MODEL}), prompt v1")
    main_model = report(evaluate(MODEL, ROUTER_PROMPT_V1))

    header(f"Experiment 3: cheap router ({ROUTER_MODEL}), prompt v2")
    v2 = report(evaluate(ROUTER_MODEL, ROUTER_PROMPT))

    header("Summary")
    print(f"   cheap router, prompt v1: {cheap:.0%}")
    print(f"   main model,   prompt v1: {main_model:.0%}")
    print(f"   cheap router, prompt v2: {v2:.0%}")


if __name__ == "__main__":
    main()
