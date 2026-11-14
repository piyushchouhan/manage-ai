"""
Lesson 0.2: Conversation history, system prompt, temperature
===============================================================

Run from backend/:
    uv run python -m lessons.l02_chat           # interactive chat
    uv run python -m lessons.l02_chat --demo    # scripted 4-turn chat (watch tokens grow)
    uv run python -m lessons.l02_chat --temps   # temperature experiment

Interactive commands:  /history  /reset  /temp 0.7  /quit

Key idea: the model is stateless (Lesson 0.1). "Memory" is just a Python list
that WE keep and RESEND in full on every call.
"""

import sys

from app.llm import MODEL, client, types

SYSTEM = (
    "You are the assistant for a warehouse company with sites in Bengaluru (BLR), "
    "Mumbai (MUM) and Delhi (DEL). Be concise: at most 3 short sentences. "
    "You do NOT have database access yet. Only use numbers the user gives you. "
    "Never invent stock, invoice or payment figures."
)


class Chat:
    """A chat session = system prompt + a growing list of messages."""

    def __init__(self, system: str = SYSTEM, temperature: float = 1.0):
        self.system = system
        self.temperature = temperature
        self.history: list[types.Content] = []  # <- THIS is the memory. Nothing else.
        self.total_tokens_billed = 0

    def send(self, text: str) -> str:
        # 1. Add the user's message to history
        self.history.append(types.Content(role="user", parts=[types.Part(text=text)]))

        # 2. Send the ENTIRE history every time
        response = client.models.generate_content(
            model=MODEL,
            contents=self.history,
            config=types.GenerateContentConfig(
                system_instruction=self.system,
                temperature=self.temperature,
            ),
        )

        # 3. Add the model's reply to history, so the next call includes it.
        #    (We append the model's own Content object rather than re-creating it
        #    from text; it can carry hidden metadata, e.g. thought signatures.)
        self.history.append(response.candidates[0].content)

        u = response.usage_metadata
        self.total_tokens_billed += u.total_token_count
        turn = len(self.history) // 2
        print(
            f"   [turn {turn} | messages sent={len(self.history) - 1} | "
            f"in={u.prompt_token_count} out={u.candidates_token_count} "
            f"thinking={u.thoughts_token_count or 0} | session total={self.total_tokens_billed}]"
        )
        return response.text.strip()

    def show_history(self):
        print(f"   --- system: {self.system[:70]}...")
        for i, msg in enumerate(self.history, 1):
            text = " ".join(p.text for p in msg.parts if p.text and not p.thought)
            print(f"   {i:>2}. {msg.role:<5} : {text[:90]}")


# ----------------------------------------------------------------------------
def interactive():
    chat = Chat()
    print(f"Chatting with {MODEL} (temperature={chat.temperature}). /history /reset /temp X /quit\n")
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/history":
            chat.show_history()
            continue
        if text == "/reset":
            chat = Chat(temperature=chat.temperature)
            print("   (history cleared: the model has forgotten everything)")
            continue
        if text.startswith("/temp"):
            chat.temperature = float(text.split()[1])
            print(f"   (temperature = {chat.temperature})")
            continue
        print(f"bot > {chat.send(text)}\n")


def demo():
    chat = Chat()
    turns = [
        "Hi, I'm Ravi. I run the Bengaluru warehouse.",
        "We have 222 padlocks in stock and the reorder level is 30.",
        "If we sell 15 a week, how many weeks until we hit the reorder level?",
        "What's my name, and which warehouse do I run?",
    ]
    for t in turns:
        print(f"you > {t}")
        print(f"bot > {chat.send(t)}\n")
    print("What gets sent on the LAST call:")
    chat.show_history()


def temps():
    prompt = "Suggest ONE short, catchy name for a warehouse assistant chatbot. Reply with only the name."
    print(f"Prompt: {prompt}\n")
    for t in [0.0, 1.0, 2.0]:
        names = []
        for _ in range(3):
            r = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=t),
            )
            names.append(r.text.strip())
        print(f"   temperature={t}: {names}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--temps" in sys.argv:
        temps()
    else:
        interactive()
