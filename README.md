# Manage AI

A learning project for building a warehouse AI assistant from scratch, covering
LLM calls, tool use, agent loops, routers, orchestration, validation and background jobs.
Plain Python comes first, and frameworks come later.

- **Learning plan:** [docs/LEARNING_PLAN.md](docs/LEARNING_PLAN.md)
- **Progress tracker:** [docs/learning-tracker.xlsx](docs/learning-tracker.xlsx)

## Stack

| Part | Tech |
|---|---|
| Backend | Python 3.12, uv, SQLAlchemy, Pydantic, `google-genai` (Gemini) |
| Database | PostgreSQL |
| Frontend | React + Vite + Tailwind + shadcn/ui |

## Setup

```bash
cp .env.example .env              # add your Gemini API key

# backend
createdb warehouse_ai
cd backend
uv sync
uv run python -m scripts.seed     # 180 days of simulated warehouse data
uv run python -m scripts.create_readonly_role   # SELECT-only DB role the AI tools use
uv run python -m scripts.enable_fuzzy_search    # pg_trgm for typo-tolerant product search
brew install pgvector && uv run python -m scripts.enable_vector_search   # embeddings for logbook search (4.6)

# frontend
cd ../frontend
npm install
npm run dev                       # http://localhost:5173
```

## Chat in the browser

Three terminals (and Redis running: `brew services start redis`):

```bash
cd backend  && uv run uvicorn app.main:app --reload --port 8000   # API: POST /api/chat
cd backend  && uv run arq lessons.l82_queue.WorkerSettings          # worker: runs investigations (needs Redis)
cd frontend && npm run dev                                         # open http://localhost:5173
```

The API runs the latest lesson agent (1.2 loop + 3.3 tools). Chats are saved in
`chat_sessions` / `chat_messages` (4.1), so a page refresh keeps the conversation. Every question is saved
as a trace in `backend/traces/`; replay with `uv run python -m lessons.l14_tracing show`.

## Lessons

Run from `backend/`:

```bash
uv run python -m lessons.l01_hello        # 0.1 first Gemini call
uv run python -m lessons.l02_chat --demo  # 0.2 history, tokens, temperature
uv run python -m lessons.l03_structured   # 0.3 structured output
uv run python -m lessons.l11_function_call # 1.1 function calling (no execution)
uv run python -m lessons.l12_agent_loop    # 1.2 the agent loop
uv run python -m lessons.l13_tools         # 1.3 multiple tools and chaining
uv run python -m lessons.l14_tracing       # 1.4 tracing (then: list / show <file>)
uv run python -m lessons.l21_permissions   # 2.1 permission-scoped tools
uv run python -m lessons.l22_tool_design   # 2.2 good tool design
uv run python -m lessons.l23_injection     # 2.3 prompt injection + read-only DB role
uv run python -m lessons.l31_fuzzy_search  # 3.1 fuzzy product search (pg_trgm)
uv run python -m lessons.l32_dates         # 3.2 dates: model describes, code calculates
uv run python -m lessons.l33_clarify       # 3.3 clarifying questions instead of guessing
uv run python -m lessons.l41_history       # 4.1 chat history saved in the DB
uv run python -m lessons.l42_selections    # 4.2 'the 2nd one', 'all of them', pending choices
uv run python -m lessons.l43_compaction    # 4.3 running summary keeps long chats bounded
uv run python -m lessons.l44_budget        # 4.4 per-part token breakdown + trimming
uv run python -m lessons.l45_selection     # 4.5 send only the tools/history a question needs
uv run python -m lessons.l46_memory        # 4.6 long-term memory + retrieval (pgvector)
uv run python -m lessons.l46b_memory_injection # 4.6b poisoned note vs remember(): memory guard
uv run python -m lessons.l47_isolation     # 4.7 per-user / per-session isolation, parallel test
uv run python -m lessons.l48_context_eval  # 4.8 tests on the assembled context: needed / forbidden / budget
uv run python -m lessons.l51_router        # 5.1 intent classifier {reason, route, confidence}
uv run python -m lessons.l52_dispatch      # 5.2 route -> handler, cheap router model, fallback
uv run python -m lessons.l53_router_evals  # 5.3 router accuracy + confusion matrix (evals/router.jsonl)
uv run python -m lessons.l61_typed_tools   # 6.1 Pydantic args/results for every tool
uv run python -m lessons.l62_verify        # 6.2 structured answer, every fact re-checked against the DB
uv run python -m lessons.l63_rules         # 6.3 business rules, one retry, safe failure
uv run python -m lessons.l71_planner       # 7.1 planner: question -> checked 3-5 step plan
uv run python -m lessons.l72_workers       # 7.2 workers + synthesizer: every fact cited to step and tool
uv run python -m lessons.l73_investigation # 7.3 router -> planner -> workers -> validator; 3 planted stories
uv run python -m lessons.l81_jobs          # 8.1 investigations as background jobs: {job_id} now, poll /api/jobs/{id}
uv run python -m lessons.l82_queue         # 8.2 Redis + arq worker process: jobs survive an API restart, with progress
uv run python -m lessons.l83_streaming     # 8.3 SSE: answer tokens streamed, job progress pushed (no polling)
uv run python -m lessons.l91_response      # 9.1 response = {summary, blocks, follow_ups}; code fills blocks from tool results
uv run python -m lessons.l92_rendering     # 9.2 chat replies get tables/metrics/follow-ups; clicking a row asks about it
uv run python -m lessons.l101_langgraph    # 10.1 M7 as a LangGraph state graph; same 3/3 eval as raw
uv run python -m lessons.l102_checkpoints  # 10.2 pause before a costly step, checkpoint in Postgres, resume in a new process
uv run python -m lessons.l111_evals        # 11.1 golden set (25 questions): code checks + LLM judge -> scorecard. Run before every change.
uv run python -m lessons.l112_observability  # 11.2 cost + latency per route from llm_calls/agent_runs (--traffic: run the 25 first)
```

## Data

The seeder simulates warehouses (BLR / MUM / DEL), products, stock movements, invoices,
payments and users with per-warehouse access. The data is internally consistent, for example
stock always equals the sum of its movements. It also includes a few planted anomalies for
the agent to investigate later.
