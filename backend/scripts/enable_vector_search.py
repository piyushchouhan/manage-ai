"""Vector search (pgvector) + the staff logbook for lesson 4.6. Run once from backend/ (safe to re-run):

    brew install pgvector                       # once: adds the extension to Homebrew's Postgres
    uv run python -m scripts.enable_vector_search

The seeded movement notes are only labels ("Opening stock", "To BLR"), nothing to search.
So this adds ops_notes: free-text notes staff wrote about incidents, suppliers, customers.
Each note is embedded (turned into 768 numbers) by Gemini and stored next to its text.
"""

from sqlalchemy import text

from app.db import engine  # owner connection: creates the extension and the table
from lessons.l46_memory import embed

NOTES = [  # (date, warehouse, author, text)
    ("2026-05-25", "DEL", "Neha Gupta", "Opening stock counted and verified for all racks. No discrepancies."),
    ("2026-06-03", "MUM", "Imran Shaikh", "Mineral water pallets arriving with crushed cartons. Told the supplier to use corner guards."),
    ("2026-06-20", "BLR", "Asha Rao", "Brass hardware supplier announced an 8% price increase from July."),
    ("2026-07-02", "MUM", "Imran Shaikh", "Pest control done in the food section; mineral water area closed for one day."),
    ("2026-07-15", "BLR", "Asha Rao", "Chander and Sons pays 2-3 weeks late on average. Keep their credit limit low."),
    ("2026-08-12", "MUM", "Imran Shaikh", "Monsoon roof leak above aisle 4. 20 packs of sticky notes water-damaged and scrapped. Facilities ticket raised."),
    ("2026-08-20", "MUM", "Imran Shaikh", "Roof leak fixed. Paper goods moved to the upper racks as a precaution."),
    ("2026-08-30", "BLR", "Ravi Kumar", "Wall plugs miscounted: boxes of 100 were entered as single pieces. Corrected with a cycle count adjustment."),
    ("2026-09-10", "DEL", "Priya Nair", "Customer disputes invoice INV-00558: says 2 cartons were missing on delivery. Payment follow-up on hold."),
    ("2026-09-18", "MUM", "Vikram Mehta", "A hotel chain wants a monthly standing order of 30 packs of garbage bags. Consider raising the reorder level."),
    ("2026-09-26", "BLR", "Ravi Kumar", "Cycle count on rack B3: 5 padlocks short again, third time this month. The rack is next to the loading dock door, which stays open during unloading. Asked for CCTV footage of 24-25 Sep."),
    ("2026-10-02", "DEL", "Neha Gupta", "Corrugated box shipment delayed by 10 days because of a truck strike on NH48. Customers informed."),
    ("2026-10-16", "BLR", "Asha Rao", "Security review: the Bengaluru loading dock has no camera coverage after 7pm. Small high-value items like padlocks and brass fittings should move to the caged area."),
    ("2026-10-22", "BLR", "Vikram Mehta", "Forklift 2 out of service for a week; unloading is slow and trucks wait up to 3 hours."),
    ("2026-10-28", "DEL", "Priya Nair", "Year-end audit on 15/12. All warehouses must finish cycle counts by 10/12."),
    ("2026-11-05", "BLR", "Ravi Kumar", "Padlocks still not moved to the cage, it is full of safety equipment. Another 7 written off after today's count."),
    ("2026-11-08", "DEL", "Neha Gupta", "Padlock sales in Delhi up sharply: two big orders from a housing society. Stock down to 11, requesting a transfer from BLR."),
    ("2026-11-12", "MUM", "Imran Shaikh", "Safety helmets running low. Supplier says the next batch ships after Diwali."),
]


def enable_vector_search():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ops_notes (
                id serial PRIMARY KEY, noted_on date NOT NULL, warehouse text NOT NULL,
                author text NOT NULL, body text NOT NULL, embedding vector(768) NOT NULL)"""))
        conn.execute(text("GRANT SELECT ON ops_notes TO warehouse_ai_reader"))  # the AI reads, never writes
        conn.execute(text("TRUNCATE ops_notes RESTART IDENTITY"))
        vectors = embed([n[3] for n in NOTES], "RETRIEVAL_DOCUMENT")
        for (d, wh, author, body), v in zip(NOTES, vectors):
            conn.execute(text("INSERT INTO ops_notes (noted_on, warehouse, author, body, embedding) "
                              "VALUES (:d, :wh, :a, :b, CAST(:v AS vector))"),
                         {"d": d, "wh": wh, "a": author, "b": body, "v": str(v)})
    print(f"pgvector enabled, {len(NOTES)} notes embedded into ops_notes")


if __name__ == "__main__":
    enable_vector_search()
