"""Database connection."""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def _json_safe(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def query(sql: str, **params) -> list[dict]:
    """Run raw SQL, return rows as plain dicts with JSON-safe values (no Decimal/datetime)."""
    with SessionLocal() as db:
        rows = db.execute(text(sql), params).mappings().all()
    return [{k: _json_safe(v) for k, v in r.items()} for r in rows]
