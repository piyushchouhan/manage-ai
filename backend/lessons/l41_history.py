"""
Lesson 4.1: Chat API and saved history
=======================================

Run from backend/:   uv run python -m lessons.l41_history

Until now the BROWSER kept the chat and sent all of it on every request.
Refresh the page -> everything gone.

Now the DATABASE keeps it:
    chat_sessions  (one row per conversation)
    chat_messages  (one row per user question / assistant reply)

Each turn, chat(session_id, text) does:
    1. load the earlier messages of this session from the DB
    2. run the agent with them as history (the injection point is in 1.2's run_agent)
    3. save the new question + reply

The browser now sends only {session_id, message}. The server stays stateless:
nothing lives in Python memory between requests, so a restart loses nothing.

Experiments:
  1. Two turns, saved to the DB
  2. "Restart": a follow-up that only makes sense with the saved history
  3. What exactly gets resent to the model on the next turn?
"""

from sqlalchemy import select

from app.db import SessionLocal, engine
from app.llm import MODEL, client, types
from app.models import Base, ChatMessage, ChatSession
from lessons.common import header
from lessons.l12_agent_loop import TooManySteps
from lessons.l33_clarify import CONFIG_V5, Clarification, ask


def ensure_tables():
    """Create chat_sessions / chat_messages if missing (safe to re-run)."""
    Base.metadata.create_all(engine, tables=[ChatSession.__table__, ChatMessage.__table__])


# --- The store: plain DB functions. Owner engine, because the AI's read-only role can't write. ---

def new_session() -> int:
    with SessionLocal() as db:
        s = ChatSession()
        db.add(s)
        db.commit()
        return s.id


def session_exists(session_id: int) -> bool:
    with SessionLocal() as db:
        return db.get(ChatSession, session_id) is not None


def load_messages(session_id: int) -> list[dict]:
    with SessionLocal() as db:
        rows = db.scalars(
            select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
        )
        return [{"id": m.id, "role": m.role, "content": m.content} for m in rows]


def save_turn(session_id: int, question: str, reply: str):
    # Both rows in one transaction: never a question without its reply.
    with SessionLocal() as db:
        db.add_all([
            ChatMessage(session_id=session_id, role="user", content=question),
            ChatMessage(session_id=session_id, role="assistant", content=reply),
        ])
        db.commit()


def to_history(messages: list[dict]) -> list[types.Content]:
    # We say "assistant"; Gemini says "model".
    return [
        types.Content(role="model" if m["role"] == "assistant" else "user",
                      parts=[types.Part.from_text(text=m["content"])])
        for m in messages
    ]


# --- One chat turn: load -> agent -> save. The API calls exactly this. ---

def run_turn(text: str, tracer=None, history=None, **agent_kw) -> str | Clarification:
    """3.3's agent, with the step-limit error turned into a polite reply."""
    try:
        return ask(text, tracer=tracer, history=history, **agent_kw)
    except TooManySteps:
        return "Sorry, I couldn't finish that within the step limit. Try asking a narrower question."


def as_reply(result: str | Clarification) -> str:
    return result.as_text() if isinstance(result, Clarification) else result


def chat(session_id: int, text: str, tracer=None) -> str:
    history = to_history(load_messages(session_id))
    reply = as_reply(run_turn(text, tracer=tracer, history=history))
    save_turn(session_id, text, reply)
    return reply


def main():
    ensure_tables()

    # --- Experiment 1 ---------------------------------------------------------
    header("Experiment 1: two turns, saved to the DB")
    sid = new_session()
    print(f"   new session id = {sid}")
    chat(sid, "How many padlocks do we have in Bangalore?")
    print("\n   chat_messages rows for this session:")
    for m in load_messages(sid):
        print(f"     {m['role']:<10} {m['content'][:90]!r}")

    # --- Experiment 2 ---------------------------------------------------------
    header("Experiment 2: 'restart' -> a follow-up that needs the saved history")
    print("   Nothing is kept in Python memory: chat() reads the history back from the DB.")
    print(chat(sid, "and in Mumbai?"))
    print("\n-> 'and in Mumbai?' alone means nothing. It worked because turn 1 came back from the DB.")

    # --- Experiment 3 ---------------------------------------------------------
    header("Experiment 3: what exactly gets resent on the next turn?")
    history = to_history(load_messages(sid))
    print("   On the next question, the model receives:")
    print(f"     1. system prompt     ({len(CONFIG_V5.system_instruction)} chars, same every call)")
    print(f"     2. tool declarations ({len(CONFIG_V5.tools[0].function_declarations)} tools, same every call)")
    print(f"     3. saved history     ({len(history)} messages):")
    for c in history:
        print(f"          {c.role:<6} {c.parts[0].text[:70]!r}")
    print("     4. the new question")
    tokens = client.models.count_tokens(model=MODEL, contents=history).total_tokens
    print(f"\n   saved history alone = {tokens} tokens, and it grows every turn (4.3 fixes that)")
    print("-> NOT resent: the tool calls and tool results of earlier turns. We only saved the")
    print("   final text, so old evidence is gone; the model must call tools again if it needs it.")


if __name__ == "__main__":
    main()
