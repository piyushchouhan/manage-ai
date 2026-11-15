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

# frontend
cd ../frontend
npm install
npm run dev                       # http://localhost:5173
```

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
```

## Data

The seeder simulates warehouses (BLR / MUM / DEL), products, stock movements, invoices,
payments and users with per-warehouse access. The data is internally consistent, for example
stock always equals the sum of its movements. It also includes a few planted anomalies for
the agent to investigate later.
