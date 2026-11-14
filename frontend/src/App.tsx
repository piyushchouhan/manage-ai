import { FormEvent, useEffect, useRef, useState } from "react";
import { Bot, Loader2, SendHorizontal, Warehouse } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

type Message = { role: "user" | "assistant"; content: string; error?: boolean };

const SUGGESTIONS = [
  "How many padlocks do we have in Bengaluru?",
  "Which customers have overdue invoices?",
  "What is running low on stock?",
];

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text: string) {
    const content = text.trim();
    if (!content || loading) return;

    const history = [...messages, { role: "user" as const, content }];
    setMessages(history);
    setInput("");
    setLoading(true);

    try {
      // Backend contract: POST /api/chat { messages } -> { reply }
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.filter((m) => !m.error).map(({ role, content }) => ({ role, content })),
        }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data: { reply: string } = await res.json();
      setMessages((m) => [...m, { role: "assistant", content: data.reply }]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Couldn't reach the backend (${(err as Error).message}). Is it running on :8000?`,
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    send(input);
  }

  return (
    <div className="mx-auto flex h-dvh max-w-3xl flex-col bg-background text-foreground">
      <header className="flex items-center gap-3 border-b px-4 py-3">
        <div className="flex size-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Warehouse className="size-5" />
        </div>
        <div className="flex-1">
          <h1 className="text-base font-semibold leading-tight">Manage AI</h1>
          <p className="text-xs text-muted-foreground">Warehouse assistant</p>
        </div>
        <Badge variant="secondary">Gemini</Badge>
      </header>

      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-3 p-4">
          {messages.length === 0 && (
            <Card className="mx-auto mt-16 w-full max-w-md text-center">
              <CardContent className="flex flex-col items-center gap-4">
                <Bot className="size-8 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  Ask about stock, movements, invoices or payments.
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((s) => (
                    <Button key={s} variant="outline" size="sm" onClick={() => send(s)}>
                      {s}
                    </Button>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {messages.map((m, i) => (
            <div
              key={i}
              className={cn(
                "max-w-[80%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
                m.role === "user"
                  ? "self-end rounded-br-sm bg-primary text-primary-foreground"
                  : "self-start rounded-bl-sm border bg-card text-card-foreground",
                m.error && "border-destructive/40 text-destructive"
              )}
            >
              {m.content}
            </div>
          ))}

          {loading && (
            <div className="flex items-center gap-2 self-start rounded-2xl border bg-card px-4 py-2.5 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" /> Thinking…
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <form onSubmit={onSubmit} className="flex gap-2 border-t p-4">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something…"
          className="h-10"
          autoFocus
        />
        <Button type="submit" size="icon-lg" disabled={loading || !input.trim()} aria-label="Send">
          <SendHorizontal />
        </Button>
      </form>
    </div>
  );
}
