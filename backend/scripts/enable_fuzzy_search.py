"""Enable fuzzy text search (pg_trgm) for lesson 3.1. Run once from backend/ (safe to re-run):

    uv run python -m scripts.enable_fuzzy_search
"""

from sqlalchemy import text

from app.db import engine  # owner connection: needed to create extensions and indexes


def enable_fuzzy_search():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        # Index of 3-letter chunks of every product name: keeps fuzzy search fast on big tables
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS products_name_trgm ON products USING gin (lower(name) gin_trgm_ops)"
        ))
    print("pg_trgm enabled, products_name_trgm index ready")


if __name__ == "__main__":
    enable_fuzzy_search()
