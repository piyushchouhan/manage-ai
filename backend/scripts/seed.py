"""Generate realistic warehouse data by simulating day-by-day operations.

Run from backend/:   uv run python -m scripts.seed

Guarantees (so the AI can later be validated against them):
  * stock_levels.quantity == SUM(stock_movements.quantity) per product+warehouse
  * stock never goes negative at any point in time
  * invoice.subtotal == SUM(invoice_lines.line_total), total == subtotal + tax
  * invoice.status matches SUM(payments.amount)

Planted stories for later "investigation" questions:
  * SKU in the BLR warehouse with repeated unexplained negative adjustments (shrinkage)
  * Two customers who never pay -> overdue invoices, one over its credit limit
  * One supplier with very long lead time -> its products often run low
"""

import random
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal

from faker import Faker
from sqlalchemy import func, select

from app.db import SessionLocal, engine
from app.models import (
    Base,
    Category,
    Customer,
    Invoice,
    InvoiceLine,
    Payment,
    Product,
    StockLevel,
    StockMovement,
    Supplier,
    User,
    UserWarehouseAccess,
    Warehouse,
)

SEED = 42
DAYS = 180
TODAY = date.today()
START = TODAY - timedelta(days=DAYS)
TAX_RATE = Decimal("0.18")

random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)


def money(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def at(d: date) -> datetime:
    return datetime.combine(d, time(random.randint(8, 19), random.randint(0, 59)))


# ---------------------------------------------------------------- catalogue

CATALOGUE = {
    "Packaging": [
        ("Corrugated Box 12x12x12", "pcs", 18), ("Corrugated Box 18x12x10", "pcs", 24),
        ("Bubble Wrap Roll 1m x 50m", "roll", 650), ("Stretch Film 18in", "roll", 420),
        ("BOPP Tape Brown 48mm", "roll", 35), ("BOPP Tape Clear 48mm", "roll", 35),
        ("Poly Mailer Bag 10x14", "pcs", 4),
    ],
    "Electrical": [
        ("LED Bulb 9W Cool White", "pcs", 65), ("LED Tube Light 20W", "pcs", 180),
        ("Extension Board 4 Socket", "pcs", 320), ("Copper Wire 1.5mm 90m", "roll", 1450),
        ("MCB 32A Single Pole", "pcs", 210), ("Modular Switch 6A", "pcs", 38),
    ],
    "Hardware": [
        ("Hex Bolt M10 x 50", "box", 420), ("Wood Screw 1.5in", "box", 140),
        ("Wall Plug 8mm", "box", 90), ("Door Hinge 4in Steel", "pcs", 55),
        ("Padlock 50mm Brass", "pcs", 260), ("Drill Bit Set 13pc", "pcs", 540),
    ],
    "Cleaning": [
        ("Floor Cleaner 5L", "pcs", 380), ("Hand Wash Liquid 5L", "pcs", 450),
        ("Garbage Bag Large 30pc", "pcs", 95), ("Microfiber Cloth 40x40", "pcs", 45),
        ("Toilet Cleaner 1L", "pcs", 110), ("Mop Refill Cotton", "pcs", 85),
    ],
    "Office Supplies": [
        ("A4 Copier Paper 75gsm", "box", 1350), ("Ball Pen Blue 50pc", "box", 210),
        ("Stapler No.10", "pcs", 75), ("Whiteboard Marker 10pc", "box", 160),
        ("Box File Large", "pcs", 95), ("Sticky Notes 3x3", "pcs", 40),
    ],
    "Safety": [
        ("Safety Helmet Yellow", "pcs", 220), ("Nitrile Gloves 100pc", "box", 480),
        ("Safety Shoes Size 9", "pcs", 1150), ("Reflective Vest", "pcs", 140),
        ("Ear Plug Foam 50pr", "box", 310), ("Fire Extinguisher 4kg", "pcs", 2100),
    ],
    "Food & Beverage": [
        ("Basmati Rice 25kg", "pcs", 2300), ("Sunflower Oil 15L", "pcs", 1950),
        ("Sugar 50kg", "pcs", 2150), ("Tea Leaves 1kg", "pcs", 380),
        ("Instant Coffee 200g", "pcs", 290), ("Mineral Water 1L x12", "box", 140),
    ],
    "Plumbing": [
        ("PVC Pipe 1in 3m", "pcs", 160), ("Ball Valve 1in Brass", "pcs", 340),
        ("Teflon Tape", "roll", 12), ("Water Tap Chrome", "pcs", 420),
        ("Flexible Hose 18in", "pcs", 95),
    ],
}

WAREHOUSES = [
    ("BLR", "Bengaluru Central", "Bengaluru"),
    ("MUM", "Mumbai Bhiwandi", "Mumbai"),
    ("DEL", "Delhi Okhla", "New Delhi"),
]

USERS = [  # name, role, warehouse codes (None = all)
    ("Asha Rao", "admin", None),
    ("Vikram Mehta", "manager", ["BLR", "MUM"]),
    ("Neha Gupta", "manager", ["DEL"]),
    ("Ravi Kumar", "staff", ["BLR"]),
    ("Imran Shaikh", "staff", ["MUM"]),
    ("Priya Nair", "accountant", None),
]


def seed() -> None:
    print("Resetting schema...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    s = SessionLocal()

    # ---------------- master data
    warehouses = [Warehouse(code=c, name=n, city=city) for c, n, city in WAREHOUSES]
    s.add_all(warehouses)

    categories = {name: Category(name=name) for name in CATALOGUE}
    s.add_all(categories.values())

    suppliers = []
    for i in range(6):
        suppliers.append(Supplier(
            name=fake.company(),
            email=fake.company_email(),
            phone=fake.phone_number(),
            lead_time_days=21 if i == 0 else random.randint(3, 8),  # supplier 0 = slow
        ))
    s.add_all(suppliers)

    customers = []
    for _ in range(25):
        customers.append(Customer(
            name=fake.company(),
            email=fake.company_email(),
            city=random.choice(["Bengaluru", "Mumbai", "Pune", "New Delhi", "Gurugram", "Chennai", "Hyderabad"]),
            credit_limit=money(random.choice([50_000, 100_000, 200_000, 500_000])),
            payment_terms_days=random.choice([15, 30, 45]),
        ))
    customers[0].credit_limit = money(50_000)  # will blow past this
    s.add_all(customers)
    s.flush()

    # payer behaviour profile per customer
    behaviour = {c.id: random.choices(["prompt", "late", "partial"], [6, 3, 1])[0] for c in customers}
    behaviour[customers[0].id] = "never"
    behaviour[customers[1].id] = "never"

    products = []
    n = 1
    for cat_name, items in CATALOGUE.items():
        for name, unit, cost in items:
            cost = money(cost)
            products.append(Product(
                sku=f"{cat_name[:3].upper()}-{n:04d}",
                name=name,
                category_id=categories[cat_name].id,
                supplier_id=random.choice(suppliers).id,
                unit=unit,
                cost_price=cost,
                sell_price=money(float(cost) * random.uniform(1.2, 1.6)),
                reorder_level=random.choice([20, 30, 50, 80]),
            ))
            n += 1
    s.add_all(products)

    users = []
    for name, role, _ in USERS:
        users.append(User(name=name, email=name.lower().replace(" ", ".") + "@manage-ai.local", role=role))
    s.add_all(users)
    s.flush()

    wh_by_code = {w.code: w for w in warehouses}
    for u, (_, _, codes) in zip(users, USERS):
        for code in codes or wh_by_code:
            s.add(UserWarehouseAccess(user_id=u.id, warehouse_id=wh_by_code[code].id))

    staff_by_wh = {
        wh_by_code["BLR"].id: [users[3].id, users[1].id],
        wh_by_code["MUM"].id: [users[4].id, users[1].id],
        wh_by_code["DEL"].id: [users[2].id],
    }
    supplier_by_id = {sp.id: sp for sp in suppliers}

    # ---------------- simulation state
    stock: dict[tuple[int, int], int] = defaultdict(int)
    movements: list[StockMovement] = []
    pending_po: dict[tuple[int, int], date] = {}  # (product, wh) -> arrival date
    counters = defaultdict(int)

    def next_ref(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}-{counters[prefix]:05d}"

    def move(p: Product, wh_id: int, kind: str, qty: int, d: date, ref=None, note=None):
        stock[(p.id, wh_id)] += qty
        assert stock[(p.id, wh_id)] >= 0, "stock went negative"
        movements.append(StockMovement(
            product_id=p.id, warehouse_id=wh_id, movement_type=kind, quantity=qty,
            reference=ref, note=note, created_by=random.choice(staff_by_wh[wh_id]), occurred_at=at(d),
        ))

    # which warehouses carry which products (not every product everywhere)
    carried = {
        w.id: [p for p in products if w.code == "BLR" or random.random() < 0.7]
        for w in warehouses
    }

    # opening stock
    for w in warehouses:
        for p in carried[w.id]:
            move(p, w.id, "purchase_in", random.randint(p.reorder_level * 2, p.reorder_level * 5),
                 START, next_ref("PO"), "Opening stock")

    shrink_product = next(p for p in carried[wh_by_code["BLR"].id] if p.name.startswith("Padlock"))

    invoices: list[Invoice] = []

    # ---------------- day-by-day simulation
    for day in range(1, DAYS + 1):
        d = START + timedelta(days=day)
        weekend = d.weekday() >= 5

        # 1. purchase orders arriving
        for key, arrival in list(pending_po.items()):
            if arrival == d:
                pid, wh_id = key
                p = next(x for x in products if x.id == pid)
                move(p, wh_id, "purchase_in", p.reorder_level * random.randint(3, 5), d, next_ref("PO"))
                del pending_po[key]

        # 2. sales
        for _ in range(random.randint(1, 3) if weekend else random.randint(4, 9)):
            w = random.choice(warehouses)
            cust = random.choice(customers)
            lines = []
            for p in random.sample(carried[w.id], k=random.randint(1, 4)):
                avail = stock[(p.id, w.id)]
                if avail <= 0:
                    continue
                qty = min(avail, random.randint(1, max(2, p.reorder_level // 3)))
                lines.append((p, qty))
            if not lines:
                continue

            inv = Invoice(
                number=next_ref("INV"), customer_id=cust.id, warehouse_id=w.id,
                issue_date=d, due_date=d + timedelta(days=cust.payment_terms_days), status="unpaid",
                subtotal=money(0), tax=money(0), total=money(0),
            )
            subtotal = Decimal("0")
            for p, qty in lines:
                lt = money(p.sell_price * qty)
                inv.lines.append(InvoiceLine(product_id=p.id, quantity=qty, unit_price=p.sell_price, line_total=lt))
                subtotal += lt
                move(p, w.id, "sale_out", -qty, d, inv.number)
            inv.subtotal = money(subtotal)
            inv.tax = money(subtotal * TAX_RATE)
            inv.total = inv.subtotal + inv.tax
            invoices.append(inv)

            # ~2% cancelled -> goods come back
            if random.random() < 0.02:
                inv.status = "cancelled"
                for p, qty in lines:
                    move(p, w.id, "return_in", qty, d + timedelta(days=1) if day < DAYS else d,
                         inv.number, "Invoice cancelled")

        # 3. reorder when low
        for w in warehouses:
            for p in carried[w.id]:
                key = (p.id, w.id)
                if stock[key] <= p.reorder_level and key not in pending_po:
                    lead = supplier_by_id[p.supplier_id].lead_time_days
                    pending_po[key] = d + timedelta(days=lead)

        # 4. occasional transfer between warehouses
        if random.random() < 0.15:
            src, dst = random.sample(warehouses, 2)
            shared = [p for p in carried[src.id] if p in carried[dst.id] and stock[(p.id, src.id)] > p.reorder_level * 2]
            if shared:
                p = random.choice(shared)
                qty = random.randint(5, p.reorder_level)
                ref = next_ref("TRF")
                move(p, src.id, "transfer_out", -qty, d, ref, f"To {dst.code}")
                move(p, dst.id, "transfer_in", qty, d, ref, f"From {src.code}")

        # 5. random small cycle-count adjustments
        if random.random() < 0.08:
            w = random.choice(warehouses)
            p = random.choice(carried[w.id])
            delta = random.choice([-3, -2, -1, 1, 2])
            if stock[(p.id, w.id)] + delta >= 0:
                move(p, w.id, "adjustment", delta, d, None, "Cycle count correction")

        # 6. planted shrinkage story: padlocks disappearing in BLR every ~10 days
        key = (shrink_product.id, wh_by_code["BLR"].id)
        if day % 10 == 0 and stock[key] >= 4:
            move(shrink_product, wh_by_code["BLR"].id, "adjustment", -random.randint(3, min(8, stock[key])), d, None, None)

    s.add_all(invoices)
    s.add_all(movements)
    s.flush()

    # ---------------- payments
    payments = []
    for inv in invoices:
        if inv.status == "cancelled":
            continue
        style = behaviour[inv.customer_id]
        if style == "never":
            continue
        terms = (inv.due_date - inv.issue_date).days
        if style == "prompt":
            pay_day = inv.issue_date + timedelta(days=random.randint(1, terms))
            amounts = [inv.total]
        elif style == "late":
            pay_day = inv.due_date + timedelta(days=random.randint(5, 40))
            amounts = [inv.total]
        else:  # partial
            pay_day = inv.issue_date + timedelta(days=random.randint(5, terms + 20))
            first = money(inv.total * Decimal(random.choice(["0.3", "0.5", "0.6"])))
            amounts = [first, inv.total - first] if random.random() < 0.5 else [first]

        for i, amt in enumerate(amounts):
            pd = pay_day + timedelta(days=i * random.randint(10, 25))
            if pd > TODAY:
                break
            payments.append(Payment(
                invoice_id=inv.id, amount=amt, paid_on=pd,
                method=random.choice(["bank_transfer", "bank_transfer", "upi", "cheque", "cash"]),
                reference=fake.bothify("TXN########"),
            ))

    s.add_all(payments)
    s.flush()

    paid = defaultdict(Decimal)
    for pay in payments:
        paid[pay.invoice_id] += pay.amount
    for inv in invoices:
        if inv.status == "cancelled":
            continue
        got = paid[inv.id]
        inv.status = "paid" if got >= inv.total else "partially_paid" if got > 0 else "unpaid"

    # ---------------- derived stock levels
    now = datetime.now()
    for (pid, wh_id), qty in stock.items():
        s.add(StockLevel(product_id=pid, warehouse_id=wh_id, quantity=qty, updated_at=now))

    s.commit()

    # ---------------- verification
    mismatches = s.execute(
        select(func.count()).select_from(
            select(StockLevel.id)
            .join(StockMovement, (StockMovement.product_id == StockLevel.product_id)
                  & (StockMovement.warehouse_id == StockLevel.warehouse_id))
            .group_by(StockLevel.id, StockLevel.quantity)
            .having(func.sum(StockMovement.quantity) != StockLevel.quantity)
            .subquery()
        )
    ).scalar()
    assert mismatches == 0, f"{mismatches} stock levels don't match movements"

    print("\nSeeded warehouse_ai:")
    for model in [Warehouse, Category, Supplier, Customer, Product, User, StockLevel,
                  StockMovement, Invoice, InvoiceLine, Payment]:
        print(f"  {model.__tablename__:<16} {s.scalar(select(func.count()).select_from(model)):>6}")
    print(f"\nDate range: {START} -> {TODAY}")
    print("Integrity check: stock_levels == SUM(movements)  OK")
    s.close()


if __name__ == "__main__":
    seed()
