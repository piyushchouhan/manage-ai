"""Settings, loaded once from the project-root .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.environ["GEMINI_MODEL"]
# small, fast, cheap: for routing and replies that need no data (lesson 5.2)
GEMINI_ROUTER_MODEL = os.getenv("GEMINI_ROUTER_MODEL", "gemini-3.5-flash-lite")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://localhost:5432/warehouse_ai")
# AI tools connect as this SELECT-only role (create it: uv run python -m scripts.create_readonly_role)
READONLY_DATABASE_URL = os.getenv(
    "READONLY_DATABASE_URL", "postgresql+psycopg://warehouse_ai_reader@localhost:5432/warehouse_ai"
)
