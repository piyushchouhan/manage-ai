"""Database connection."""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL, READONLY_DATABASE_URL

engine = create_engine(DATABASE_URL)  # owner: seeding, admin scripts
SessionLocal = sessionmaker(bind=engine)
readonly_engine = create_engine(READONLY_DATABASE_URL)  # AI tools: SELECT only (lesson 2.3)


def _json_safe(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def query(sql: str, **params) -> list[dict]:
    """Run raw SQL as the READ-ONLY role. Return rows as plain dicts with JSON-safe values.
    Every AI tool goes through here, so no tool can ever write, whatever the model asks."""
    with readonly_engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [{k: _json_safe(v) for k, v in r.items()} for r in rows]
