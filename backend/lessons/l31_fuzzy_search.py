"""
Lesson 3.1: Entity resolution (spelling variations)
====================================================

Run from backend/:   uv run python -m lessons.l31_fuzzy_search
(Needs pg_trgm once:  uv run python -m scripts.enable_fuzzy_search)

Users don't type product names exactly: "padlok", "pad lock", "brass lock".
1.3's search_products needs every word to appear in the name, so a single
typo finds nothing.

pg_trgm splits text into 3-letter chunks ("padlok" -> pad, adl, dlo, lok...)
and scores how many chunks two strings share, from 0 to 1. Typos only break
a few chunks, so the score stays high.

Ranking (best first):
  1.0   exact SKU            "HAR-0018"
  0.9   every word matches   "brass lock" -> Padlock 50mm Brass (1.3's rule)
  0-1   fuzzy score          "padlok"     -> Padlock 50mm Brass (0.71)

Experiments:
  1. Old search vs fuzzy search, no AI: which queries find the right product?
  2. Close scores: when is the top match safe to pick?
  3. The agent with fuzzy search: "how many padlok in bangalore?"
"""

from app.db import query
from app.llm import types
from lessons.common import header
from lessons.l12_agent_loop import run_agent
from lessons.l13_tools import MAX_STEPS, search_products
from lessons.l22_tool_design import CONFIG_V2, DECLS_V2, TOOLS_V2, safe

MIN_SCORE = 0.4  # below this, a "match" is noise (e.g. "hammer" -> Hand Wash Liquid, 0.29)


def search_products_fuzzy(query_text: str, limit: int = 5) -> dict:
    q = query_text.lower().strip()
    rows = query(
        """
        SELECT sku, name, unit, reorder_level,
               ROUND(CASE
                   WHEN lower(sku) = :q THEN 1.0                                    -- exact SKU
                   WHEN NOT EXISTS (                                                -- every word in the name
                        SELECT 1 FROM unnest(string_to_array(:q, ' ')) w
                        WHERE w <> '' AND strpos(lower(name), w) = 0) THEN 0.9
                   ELSE GREATEST(similarity(lower(name), :q),                       -- fuzzy
                                 word_similarity(:q, lower(name)))
               END::numeric, 2)::float AS score
        FROM products
        ORDER BY score DESC
        LIMIT :limit
        """,
        q=q, limit=limit,
    )
    matches = [r for r in rows if r["score"] >= MIN_SCORE]
    if not matches:
        return {"matches": [], "hint": "No product matches. Tell the user, and ask what they meant."}
    return {"matches": matches}


# --- The agent: 2.2's tools and prompt, with search_products swapped -------------

TOOLS_V3 = {**TOOLS_V2, "search_products": safe(search_products_fuzzy)}

DECLS_V3 = {**DECLS_V2, "search_products": types.FunctionDeclaration(
    name="search_products",
    description=(
        "Find products by name, even with typos or partial words ('padlok', 'corugated box'). "
        "Use this FIRST whenever the user names a product, to get its SKU. Returns the best "
        "matches with a score from 0 to 1 (1 = exact SKU, 0.9 = all words match, lower = fuzzy). "
        "If the top two scores are close, the name is ambiguous: ask the user which one they mean."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {"query_text": {"type": "string", "description": "What the user called the product."}},
        "required": ["query_text"],
    },
)}
CONFIG_V3 = CONFIG_V2.model_copy(update={"tools": [types.Tool(function_declarations=list(DECLS_V3.values()))]})


def ask(question: str, tracer=None, history=None) -> str:
    return run_agent(question, config=CONFIG_V3, registry=TOOLS_V3, max_steps=MAX_STEPS,
                     tracer=tracer, history=history)


def main():
    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: old search (1.3) vs fuzzy search, no AI")
    print(f"   {'query':<16}{'1.3 (all words)':<26}fuzzy (top match, score)")
    for q in ["padlok", "brass lock", "PAD lock 50", "padlocks", "corugated box", "ball valv", "HAR-0018", "hammer"]:
        old = search_products(q)["matches"]
        new = search_products_fuzzy(q)["matches"]
        old_s = old[0]["name"] if old else "-- nothing --"
        new_s = f"{new[0]['name']} ({new[0]['score']})" if new else "-- nothing --"
        print(f"   {q:<16}{old_s[:24]:<26}{new_s}")
    print("\n-> One typo kills the old search. Fuzzy survives typos, plurals and split words,")
    print("   and still returns nothing for a product we don't sell ('hammer').")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: close scores = ambiguous")
    for q in ["box", "brass", "corugated box"]:
        print(f"\n   '{q}':")
        for m in search_products_fuzzy(q)["matches"]:
            print(f"      {m['score']:<5} {m['sku']}  {m['name']}")
    print("\n-> 'box' matches 3 products equally. Picking the first would be a coin toss.")
    print("   Scores tell the model WHEN to ask instead of guess (lesson 3.3 builds on this).")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: the agent with fuzzy search")
    for q in ["how many padlok in bangalore?", "stock of corugated box in delhi"]:
        print(ask(q))


if __name__ == "__main__":
    main()
