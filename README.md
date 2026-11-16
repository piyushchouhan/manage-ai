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

# frontend
cd ../frontend
npm install
npm run dev                       # http://localhost:5173
```

## Chat in the browser

Two terminals:

```bash
cd backend  && uv run uvicorn app.main:app --reload --port 8000   # API: POST /api/chat
cd frontend && npm run dev                                         # open http://localhost:5173
```

The API runs the latest lesson agent (1.2 loop + 3.3 tools). Every question is saved
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
```

## Data

The seeder simulates warehouses (BLR / MUM / DEL), products, stock movements, invoices,
payments and users with per-warehouse access. The data is internally consistent, for example
stock always equals the sum of its movements. It also includes a few planted anomalies for
the agent to investigate later.
