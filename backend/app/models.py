"""Warehouse data model.

    warehouses ──< stock_levels >── products >── categories
        │                              │    >── suppliers
        └──────< stock_movements >─────┘
                       │
    customers ──< invoices ──< invoice_lines >── products
                     └──< payments

    users ──< user_warehouse_access >── warehouses   (permission scoping for AI tools)

    chat_sessions ──< chat_messages                  (saved chat history, lesson 4.1)
                 ──  chat_state                     (last list shown + pending question, 4.2)
                 ──  chat_summaries                 (running summary of old turns, 4.3)
    users ──< user_memories                          (long-term facts about a user, 4.6)
    chat_sessions ──< jobs                           (investigations run in the background, 8.1)
    agent_runs ──< llm_calls                         (every model call: tokens, latency, cost, 11.2)
    ops_notes  (staff logbook + pgvector embeddings, 4.6; created by scripts/enable_vector_search.py)
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    JSON,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------- Master data ----------

class Warehouse(Base):
    __tablename__ = "warehouses"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(10), unique=True)
    name: Mapped[str] = mapped_column(String(100))
    city: Mapped[str] = mapped_column(String(60))


class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(60), unique=True)


class Supplier(Base):
    __tablename__ = "suppliers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(120))
    phone: Mapped[str] = mapped_column(String(40))
    lead_time_days: Mapped[int] = mapped_column(Integer)


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(120))
    city: Mapped[str] = mapped_column(String(60))
    credit_limit: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    payment_terms_days: Mapped[int] = mapped_column(Integer)


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(String(20), unique=True)
    name: Mapped[str] = mapped_column(String(150))
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"))
    unit: Mapped[str] = mapped_column(String(10))  # pcs, box, kg, roll
    cost_price: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    sell_price: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    reorder_level: Mapped[int] = mapped_column(Integer)

    category: Mapped[Category] = relationship()
    supplier: Mapped[Supplier] = relationship()


# ---------- Inventory ----------

class StockLevel(Base):
    """Current on-hand quantity per product per warehouse (derived from movements)."""

    __tablename__ = "stock_levels"
    __table_args__ = (UniqueConstraint("product_id", "warehouse_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"))
    quantity: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class StockMovement(Base):
    """Every change in stock. quantity is signed: + in, - out."""

    __tablename__ = "stock_movements"
    __table_args__ = (
        CheckConstraint(
            "movement_type IN ('purchase_in','sale_out','transfer_in','transfer_out','adjustment','return_in')"
        ),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"), index=True)
    movement_type: Mapped[str] = mapped_column(String(20))
    quantity: Mapped[int] = mapped_column(Integer)
    reference: Mapped[str | None] = mapped_column(String(40))  # e.g. INV-00012, PO-00003, TRF-00007
    note: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)


# ---------- Sales & money ----------

class Invoice(Base):
    __tablename__ = "invoices"
    __table_args__ = (CheckConstraint("status IN ('paid','partially_paid','unpaid','cancelled')"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    number: Mapped[str] = mapped_column(String(20), unique=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"))
    issue_date: Mapped[date] = mapped_column(Date)
    due_date: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20))
    subtotal: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    tax: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    total: Mapped[Decimal] = mapped_column(Numeric(12, 2))

    customer: Mapped[Customer] = relationship()
    lines: Mapped[list["InvoiceLine"]] = relationship(back_populates="invoice")
    payments: Mapped[list["Payment"]] = relationship(back_populates="invoice")


class InvoiceLine(Base):
    __tablename__ = "invoice_lines"
    id: Mapped[int] = mapped_column(primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    line_total: Mapped[Decimal] = mapped_column(Numeric(12, 2))

    invoice: Mapped[Invoice] = relationship(back_populates="lines")
    product: Mapped[Product] = relationship()


class Payment(Base):
    __tablename__ = "payments"
    id: Mapped[int] = mapped_column(primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    method: Mapped[str] = mapped_column(String(20))  # bank_transfer, upi, cash, cheque
    paid_on: Mapped[date] = mapped_column(Date)
    reference: Mapped[str | None] = mapped_column(String(40))

    invoice: Mapped[Invoice] = relationship(back_populates="payments")


# ---------- Users & permissions (used later to scope AI tools) ----------

class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('admin','manager','staff','accountant')"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    email: Mapped[str] = mapped_column(String(120), unique=True)
    role: Mapped[str] = mapped_column(String(20))


class UserWarehouseAccess(Base):
    __tablename__ = "user_warehouse_access"
    __table_args__ = (UniqueConstraint("user_id", "warehouse_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"))


# ---------- Chat history (lesson 4.1) ----------

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))  # owner (lesson 4.7)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (CheckConstraint("role IN ('user','assistant')"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    display: Mapped[dict | None] = mapped_column(JSON)  # 9.2: {blocks, follow_ups} shown under a reply
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatState(Base):
    """What the user is looking at (lesson 4.2): the last numbered list, and any open question."""
    __tablename__ = "chat_state"
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), primary_key=True)
    selection: Mapped[list] = mapped_column(JSON, default=list)  # [{"sku", "name"}, ...] in shown order
    pending: Mapped[dict | None] = mapped_column(JSON)  # {"question", "asked", "options"} or None


class ChatSummary(Base):
    """Running summary of a long chat (lesson 4.3). Messages with id <= upto_id are folded into it."""
    __tablename__ = "chat_summaries"
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    upto_id: Mapped[int] = mapped_column(Integer, default=0)


class UserMemory(Base):
    """Long-term memory (lesson 4.6): lasting facts about a user, kept across chat sessions."""
    __tablename__ = "user_memories"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    fact: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------- Background jobs (lesson 8.1) ----------

class Job(Base):
    """One investigation, run after /api/chat has already answered with its id. The UI polls it."""
    __tablename__ = "jobs"
    __table_args__ = (CheckConstraint("status IN ('queued','running','done','failed')"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)  # only the owner may poll it
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"))
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    reply: Mapped[str | None] = mapped_column(Text)
    progress: Mapped[str | None] = mapped_column(Text)  # "step 2/3: checking movements" (8.2)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


# ---------- Observability (lesson 11.2) ----------

class AgentRun(Base):
    """One request through the assistant (a chat turn, or an investigation job), with its route."""
    __tablename__ = "agent_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    route: Mapped[str] = mapped_column(String(40), index=True)
    user_id: Mapped[int | None] = mapped_column(Integer)
    question: Mapped[str] = mapped_column(Text)
    seconds: Mapped[float] = mapped_column(Float)
    ok: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class LlmCall(Base):
    """One model call. run_id is NULL for calls outside a request (e.g. the eval's judge)."""
    __tablename__ = "llm_calls"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(32), index=True)
    model: Mapped[str] = mapped_column(String(60))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    thinking_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))  # NULL = no price known for this model
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
