"""
Lesson 1.4: Tracing
====================

Run from backend/:
    uv run python -m lessons.l14_tracing                  # run 2 questions, save + show their traces
    uv run python -m lessons.l14_tracing list             # list saved traces
    uv run python -m lessons.l14_tracing show <file>      # replay one trace (no AI call, no DB)

An agent makes several model calls and tool calls per question. When an answer
is wrong, slow or expensive, you need to know WHICH step caused it. Printing to
the terminal is gone once you close it. A TRACE is a file with one JSON line per
event, so any question can be replayed and inspected later.

    {"event": "model_call", "step": 1, "latency_ms": 812, "prompt_tokens": 603, ...}
    {"event": "tool_call",  "step": 1, "tool": "search_products", "args": {...}, ...}

Same loop as 1.2, same tools as 1.3. We only pass a Tracer into it.

Experiments:
  1. Trace a simple question and an investigation, then replay both from their files
  2. Read the summary: which step was slowest, and where did the tokens go?
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from lessons.common import header
from lessons.l13_tools import ask

TRACE_DIR = Path(__file__).resolve().parents[1] / "traces"  # backend/traces/ (git-ignored)


# --- The tracer: run_agent calls these three methods -----------------------------

class Tracer:
    def __init__(self, question: str):
        TRACE_DIR.mkdir(exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:40]
        self.path = TRACE_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{slug}.jsonl"
        self.started = time.perf_counter()
        self._write(event="start", question=question)

    def _write(self, **event):
        # One JSON object per line ("JSON Lines"). Append-only: a crash mid-run still leaves a usable file.
        event["at"] = datetime.now().isoformat(timespec="milliseconds")
        with self.path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def model_call(self, step, response, seconds):
        u = response.usage_metadata
        calls = response.function_calls or []
        self._write(
            event="model_call", step=step, model=response.model_version, latency_ms=round(seconds * 1000),
            prompt_tokens=u.prompt_token_count, output_tokens=u.candidates_token_count or 0,
            thinking_tokens=u.thoughts_token_count or 0,
            decision=[c.name for c in calls] or "final_answer",
        )

    def tool_call(self, step, name, args, result, seconds):
        self._write(
            event="tool_call", step=step, tool=name, args=dict(args), latency_ms=round(seconds * 1000),
            result_chars=len(json.dumps(result, default=str)),  # big results = big prompts next step
            result=result,
        )

    def finish(self, answer=None, error=None):
        self._write(event="finish", total_ms=round((time.perf_counter() - self.started) * 1000),
                    answer=answer, error=error)


# --- Replay: rebuild the story from the file alone --------------------------------

def show(path: Path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    start = events[0]
    print(f"\nTRACE  {path.name}")
    print(f"Q:     {start['question']}\n")
    print(f"  {'step':<5}{'what':<52}{'ms':>7}{'prompt tok':>12}{'out tok':>9}")
    print(f"  {'-' * 84}")

    model_calls = [e for e in events if e["event"] == "model_call"]
    tool_calls = [e for e in events if e["event"] == "tool_call"]
    for e in events:
        if e["event"] == "model_call":
            what = "model -> " + (", ".join(e["decision"]) if isinstance(e["decision"], list) else "FINAL ANSWER")
            out = e["output_tokens"] + e["thinking_tokens"]
            print(f"  {e['step']:<5}{what[:51]:<52}{e['latency_ms']:>7}{e['prompt_tokens']:>12}{out:>9}")
        elif e["event"] == "tool_call":
            args = ", ".join(f"{k}={v}" for k, v in e["args"].items())
            what = f"  code  -> {e['tool']}({args})"
            print(f"  {'':<5}{what[:51]:<52}{e['latency_ms']:>7}   result: {e['result_chars']:,} chars")
        elif e["event"] == "finish":
            end = e

    # The summary answers the self-check questions
    slowest = max(model_calls + tool_calls, key=lambda e: e["latency_ms"])
    biggest = max(model_calls, key=lambda e: e["prompt_tokens"])
    total_in = sum(e["prompt_tokens"] for e in model_calls)
    total_out = sum(e["output_tokens"] + e["thinking_tokens"] for e in model_calls)
    model_ms = sum(e["latency_ms"] for e in model_calls)
    tool_ms = sum(e["latency_ms"] for e in tool_calls)
    label = lambda e: f"step {e['step']} " + (f"tool {e['tool']}" if e["event"] == "tool_call" else "model call")

    print(f"  {'-' * 84}")
    print(f"  model calls : {len(model_calls)}    tool calls: {len(tool_calls)}    total: {end['total_ms']:,} ms "
          f"(model {model_ms:,} / tools {tool_ms:,})")
    print(f"  tokens      : {total_in:,} in + {total_out:,} out  (every call re-sends the whole history)")
    print(f"  slowest     : {label(slowest)}  ({slowest['latency_ms']:,} ms)")
    print(f"  most tokens : {label(biggest)}  ({biggest['prompt_tokens']:,} prompt tokens)")
    if end.get("error"):
        print(f"  ERROR       : {end['error']}")
    else:
        print(f"  answer      : {end['answer'][:160].replace(chr(10), ' ')}...")


def traced(question: str) -> Path:
    tracer = Tracer(question)
    ask(question, tracer=tracer)
    return tracer.path


def main():
    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: run two questions WITH a tracer (live output as before)")
    paths = [
        traced("How many brass locks do we have in Bangalore?"),
        traced("Why is padlock stock in BLR dropping?"),
    ]

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: replay both traces from their files (no AI, no DB)")
    for p in paths:
        show(p)
    print("\n-> Compare the 'prompt tok' column step by step. What makes it jump?")
    print(f"   Files are in {TRACE_DIR}/ . Replay any of them later with:  show <file>")


if __name__ == "__main__":
    cmd = sys.argv[1:] or ["run"]
    if cmd[0] == "list":
        for p in sorted(TRACE_DIR.glob("*.jsonl")):
            print(p.name)
    elif cmd[0] == "show":
        target = Path(cmd[1]) if len(cmd) > 1 else max(TRACE_DIR.glob("*.jsonl"))  # default: latest
        show(target if target.exists() else TRACE_DIR / target)
    else:
        main()
