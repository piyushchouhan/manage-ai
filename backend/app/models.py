"""Warehouse data model.

    warehouses ──< stock_levels >── products >── categories
        │                              │    >── suppliers
        └──────< stock_movements >─────┘
                       │
    customers ──< invoices ──< invoice_lines >── products
                     └──< payments

    users ──< user_warehouse_access >── warehouses   (permission scoping for AI tools)
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
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
