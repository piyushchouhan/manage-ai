import { FormEvent, Fragment, useEffect, useRef, useState } from "react";
import { Bot, Loader2, Plus, SendHorizontal, Warehouse } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Blocks, type Display } from "@/components/Blocks";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

type Message = {
  role: "user" | "assistant";
  content: string;
  error?: boolean;
  display?: Display | null;
};

const SUGGESTIONS = [
  "How many padlocks do we have in Bengaluru?",
  "Which customers have overdue invoices?",
  "What is running low on stock?",
];

type User = { id: number; name: string; role: string; warehouses: string[] };

// The chat itself lives in the DB (lesson 4.1). The browser only remembers which user it
// is acting as, and that user's current session (each user has their own, lesson 4.7).
const USER_KEY = "manage-ai-user";
const sessionKey = (userId: number) => `manage-ai-session-${userId}`;
const jobKey = (userId: number) => `manage-ai-job-${userId}`; // a running investigation (8.1), survives a refresh
function read(key: string): number | null {
  try {
    const v = localStorage.getItem(key);
    return v ? Number(v) : null;
  } catch {
    return null;
  }
}
function write(key: string, id: number | null) {
  try {
    if (id === null) localStorage.removeItem(key);
    else localStorage.setItem(key, String(id));
  } catch {
    /* private window etc.: the chat still works, it just won't survive a refresh */
  }
}

// Investigations run as background jobs (8.1) in a worker (8.2); their progress is pushed over SSE (8.3).
type Job = {
  id: number;
  status: "queued" | "running" | "done" | "failed";
  progress: string | null; // "step 2/3: ..." written by the worker (lesson 8.2)
  reply: string | null;
  question: string;
};

// Server-Sent Events over fetch (EventSource can't POST or send our X-User-Id header).
// The body arrives in pieces; events are separated by a blank line: "event: token\ndata: {...}\n\n".
async function readSSE(res: Response, onEvent: (event: string, data: any) => void) {
  if (!res.ok || !res.body) throw new Error(`Server returned ${res.status}`);
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const event = block.match(/^event: (.*)$/m)?.[1];
      const data = block.match(/^data: (.*)$/m)?.[1];
      if (event && data) onEvent(event, JSON.parse(data)); // lines starting with ":" are keep-alives
    }
  }
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [investigatingSince, setInvestigatingSince] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const [progress, setProgress] = useState<string | null>(null);
  const [draft, setDraft] = useState<string | null>(null); // the answer so far, while it streams
  const [lookingUp, setLookingUp] = useState<string[]>([]); // tools the agent is using right now
  const [users, setUsers] = useState<User[]>([]);
  // Stand-in for a login: the backend trusts this id (X-User-Id). Not secure, see backend/app/main.py.
  const [userId, setUserId] = useState<number>(() => read(USER_KEY) ?? 1);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const headers = {
    "Content-Type": "application/json",
    "X-User-Id": String(userId),
  };

  useEffect(() => {
    fetch("/api/users")
      .then((res) => res.json())
      .then(setUsers)
      .catch(() => {});
  }, []);

  // On load, refresh, or user switch: fetch THIS user's saved chat from the backend
  useEffect(() => {
    write(USER_KEY, userId);
    const id = read(sessionKey(userId));
    setSessionId(id);
    setMessages([]);
    if (id === null) return;
    fetch(`/api/sessions/${id}/messages`, {
      headers: { "X-User-Id": String(userId) },
    })
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((saved: Message[]) => {
        setMessages(saved);
        const job = read(jobKey(userId));
        if (job !== null) resume(job); // refreshed mid-investigation: keep waiting for it
      })
      .catch(() => {
        write(sessionKey(userId), null); // gone, or not this user's: start fresh
        setSessionId(null);
      });
  }, [userId]);

  // Tick once a second while investigating, for the elapsed time
  useEffect(() => {
    if (investigatingSince === null) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [investigatingSince]);

  async function waitFor(jobId: number): Promise<Job> {
    write(jobKey(userId), jobId);
    setInvestigatingSince((s) => s ?? Date.now());
    try {
      let job: Job | null = null;
      const res = await fetch(`/api/jobs/${jobId}/events`, {
        headers: { "X-User-Id": String(userId) },
      });
      await readSSE(res, (_event, data: Job) => {
        job = data;
        setProgress(data.status === "queued" ? "waiting for a worker" : data.progress);
      });
      const last = job as Job | null;
      if (!last || (last.status !== "done" && last.status !== "failed"))
        throw new Error("the stream ended early");
      return last;
    } finally {
      write(jobKey(userId), null);
      setInvestigatingSince(null);
      setProgress(null);
    }
  }

  async function resume(jobId: number) {
    setLoading(true);
    try {
      const job = await waitFor(jobId);
      // finished while we were away? then the reload above already has it
      setMessages((m) =>
        m[m.length - 1]?.content === job.reply
          ? m
          : [...m, { role: "user", content: job.question }, { role: "assistant", content: job.reply ?? "" }],
      );
    } catch {
      /* gone: nothing to resume */
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    write(sessionKey(userId), null);
    setSessionId(null);
    setMessages([]);
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text: string) {
    const content = text.trim();
    if (!content || loading) return;

    setMessages((m) => [...m, { role: "user", content }]);
    setInput("");
    setLoading(true);

    try {
      // Backend contract (8.3): POST /api/chat/stream { session_id, message } -> SSE events
      //   session {session_id} · tool {names} · token {text}... · then done {reply} or job {job_id}
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers,
        body: JSON.stringify({ session_id: sessionId, message: content }),
      });
      let reply: string | null = null;
      let jobId: number | null = null;
      let display: Display | null = null; // 9.2: tables etc., built from the tool results
      await readSSE(res, (event, data) => {
        if (event === "session") {
          write(sessionKey(userId), data.session_id);
          setSessionId(data.session_id);
        } else if (event === "tool") {
          setLookingUp(data.names);
          setDraft(null); // text before a tool call was the model thinking aloud, not the answer
        } else if (event === "token") {
          setLookingUp([]);
          setDraft((d) => (d ?? "") + data.text);
        } else if (event === "display") display = data;
        else if (event === "done") reply = data.reply;
        else if (event === "job") jobId = data.job_id;
        else if (event === "error") throw new Error(data.message);
      });
      if (jobId !== null) reply = (await waitFor(jobId)).reply;
      if (reply === null) throw new Error("the stream ended early");
      const final: string = reply;
      const shown: Display | null = display;
      setMessages((m) => [...m, { role: "assistant", content: final, display: shown }]);
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
      setDraft(null);
      setLookingUp([]);
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
        <select
          value={userId}
          onChange={(e) => setUserId(Number(e.target.value))}
          disabled={loading}
          aria-label="Acting as user"
          className="h-8 rounded-md border bg-background px-2 text-sm"
        >
          {users.map((u) => (
            <option key={u.id} value={u.id}>
              {u.name} ({u.warehouses.join(", ")})
            </option>
          ))}
        </select>
        <Button variant="ghost" size="sm" onClick={newChat} disabled={loading}>
          <Plus /> New chat
        </Button>
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
            <Fragment key={i}>
              <div
                className={cn(
                  "max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
                  m.role === "user"
                    ? "self-end whitespace-pre-wrap rounded-br-sm bg-primary text-primary-foreground"
                    : "markdown self-start overflow-x-auto rounded-bl-sm border bg-card text-card-foreground",
                  m.error && "border-destructive/40 text-destructive",
                )}
              >
                {/* The model answers in markdown (bold, lists, tables), so render it */}
                {m.role === "assistant" ? (
                  <Markdown remarkPlugins={[remarkGfm]}>{m.content}</Markdown>
                ) : (
                  m.content
                )}
              </div>
              {m.display && <Blocks display={m.display} onAsk={send} disabled={loading} />}
            </Fragment>
          ))}

          {loading && draft !== null && (
            <div className="markdown max-w-[80%] self-start overflow-x-auto rounded-2xl rounded-bl-sm border bg-card px-4 py-2.5 text-sm leading-relaxed text-card-foreground">
              <Markdown remarkPlugins={[remarkGfm]}>{draft}</Markdown>
            </div>
          )}

          {loading && draft === null && (
            <div className="flex items-center gap-2 self-start rounded-2xl border bg-card px-4 py-2.5 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              {investigatingSince === null ? (
                lookingUp.length ? (
                  `Looking up ${lookingUp.join(", ")}…`
                ) : (
                  "Thinking…"
                )
              ) : (
                <span>
                  Investigating… {Math.max(0, Math.round((now - investigatingSince) / 1000))}s
                  <span className="block text-xs">{progress ?? "starting"} · usually 1–2 minutes</span>
                </span>
              )}
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
