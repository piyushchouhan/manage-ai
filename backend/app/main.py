"""Chat API for the React frontend.

Run from backend/:   uv run uvicorn app.main:app --reload --port 8000
Then the frontend:   cd frontend && npm run dev   ->  http://localhost:5173

Contract (what frontend/src/App.tsx sends and expects):
    POST /api/chat   {"messages": [{"role": "user"|"assistant", "content": "..."}, ...]}
                  -> {"reply": "..."}

The agent is the latest one from the lessons: 1.2's loop + 3.1's tools.
When a later lesson improves the agent, change the imports below.

Not yet (coming in later lessons): login/permissions (2.1 is ready, needs a
user from the session), saved chat history (4.1), streaming (8.3).
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app.llm import types
from lessons.l12_agent_loop import TooManySteps, run_agent
from lessons.l14_tracing import Tracer
from lessons.l31_fuzzy_search import CONFIG_V3, TOOLS_V3

MAX_STEPS = 8

app = FastAPI(title="Manage AI")


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class ChatReply(BaseModel):
    reply: str


def to_history(messages: list[ChatMessage]) -> list[types.Content]:
    # The browser calls the AI "assistant"; Gemini calls it "model".
    return [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part.from_text(text=m.content)],
        )
        for m in messages
    ]


@app.post("/api/chat", response_model=ChatReply)
def chat(req: ChatRequest) -> ChatReply:
    # Plain `def` (not async): FastAPI runs it in a thread, so a slow model call
    # doesn't block other requests.
    *earlier, last = req.messages
    try:
        reply = run_agent(
            last.content,
            config=CONFIG_V3,
            registry=TOOLS_V3,
            max_steps=MAX_STEPS,
            tracer=Tracer(last.content),  # every browser question is saved to backend/traces/
            history=to_history(earlier),
        )
    except TooManySteps:
        reply = "Sorry, I couldn't finish that within the step limit. Try asking a narrower question."
    return ChatReply(reply=reply)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
