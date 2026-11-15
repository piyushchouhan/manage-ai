"""Create the read-only Postgres role that AI tools connect with (lesson 2.3).

Run once from backend/ (safe to re-run):   uv run python -m scripts.create_readonly_role

The seeder and migrations keep using your normal (owner) role. Tools use this
one, so even a buggy or tricked tool can only SELECT.
"""

from sqlalchemy import text

from app.db import engine  # owner connection: allowed to create roles and grant

ROLE = "warehouse_ai_reader"


def ensure_readonly_role():
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": ROLE}).first()
        if not exists:
            conn.execute(text(f"CREATE ROLE {ROLE} LOGIN"))
        db = conn.execute(text("SELECT current_database()")).scalar()
        for sql in [
            f'GRANT CONNECT ON DATABASE "{db}" TO {ROLE}',
            f"GRANT USAGE ON SCHEMA public TO {ROLE}",
            f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {ROLE}",
            # tables created later get SELECT too
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {ROLE}",
        ]:
            conn.execute(text(sql))
    print(f"role {ROLE}: SELECT only, ready")


if __name__ == "__main__":
    ensure_readonly_role()
