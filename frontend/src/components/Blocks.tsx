// Lesson 9.2: the blocks under a reply. Every value here was copied by the backend from a tool
// result (9.1's build), never written by the model. Clicking a product row or chip asks about it.
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

export type Option = { sku: string; name?: string; warehouse?: string };
export type Block =
  | {
      type: "table";
      title: string;
      columns: string[];
      rows: Record<string, unknown>[];
      source: string;
      options: Option[]; // options[i] = the product in row i (empty if the rows aren't products)
    }
  | { type: "metric"; label: string; value: number | string; source: string }
  | { type: "selection"; title: string; options: Option[]; source: string };
export type Display = { blocks: Block[]; follow_ups: string[] };

// Same text as backend lessons/l92_rendering.py click(): the SKU makes it unambiguous
export function askAbout(o: Option): string {
  return `Tell me more about ${o.sku} ${o.name ?? ""}${o.warehouse ? ` in ${o.warehouse}` : ""}`.trim();
}

const heading = (col: string) => col.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const show = (v: unknown) =>
  v === null || v === undefined ? "—" : typeof v === "number" ? v.toLocaleString("en-IN") : String(v);

function Source({ text }: { text: string }) {
  return <p className="text-xs text-muted-foreground">from {text}</p>;
}

export function Blocks({
  display,
  onAsk,
  disabled,
}: {
  display: Display;
  onAsk: (question: string) => void;
  disabled: boolean;
}) {
  const metrics = display.blocks.filter((b) => b.type === "metric");
  const others = display.blocks.filter((b) => b.type !== "metric");

  return (
    <div className="flex w-full max-w-[80%] flex-col gap-3 self-start">
      {metrics.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {metrics.map((m, i) => (
            <div key={i} className="min-w-32 rounded-xl border bg-card px-4 py-2.5">
              <p className="text-xs text-muted-foreground">{m.label}</p>
              <p className="text-2xl font-semibold tabular-nums">{show(m.value)}</p>
              <Source text={m.source} />
            </div>
          ))}
        </div>
      )}

      {others.map((b, i) =>
        b.type === "table" ? (
          <div key={i} className="flex flex-col gap-1.5 rounded-xl border bg-card p-3">
            <p className="text-sm font-medium">{b.title}</p>
            <Table>
              <TableHeader>
                <TableRow>
                  {b.columns.map((c) => (
                    <TableHead key={c}>{heading(c)}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {b.rows.map((row, r) => {
                  const option = b.options[r];
                  return (
                    <TableRow
                      key={r}
                      onClick={option && !disabled ? () => onAsk(askAbout(option)) : undefined}
                      title={option ? `Ask about ${option.sku}` : undefined}
                      className={cn(option && !disabled && "cursor-pointer")}
                    >
                      {b.columns.map((c) => (
                        <TableCell
                          key={c}
                          className={cn(typeof row[c] === "number" && "text-right tabular-nums")}
                        >
                          {show(row[c])}
                        </TableCell>
                      ))}
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
            <Source text={b.options.length ? `${b.source} · click a row to ask about it` : b.source} />
          </div>
        ) : b.type === "selection" ? (
          <div key={i} className="flex flex-col gap-1.5">
            <p className="text-sm font-medium">{b.title}</p>
            <div className="flex flex-wrap gap-2">
              {b.options.map((o) => (
                <Button
                  key={o.sku + (o.warehouse ?? "")}
                  variant="outline"
                  size="sm"
                  disabled={disabled}
                  onClick={() => onAsk(askAbout(o))}
                >
                  {o.sku} {o.name}
                </Button>
              ))}
            </div>
          </div>
        ) : null,
      )}

      {display.follow_ups.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {display.follow_ups.map((q) => (
            <Button
              key={q}
              variant="secondary"
              size="sm"
              disabled={disabled}
              onClick={() => onAsk(q)}
              className="h-auto whitespace-normal py-1.5 text-left"
            >
              {q}
            </Button>
          ))}
        </div>
      )}
    </div>
  );
}
