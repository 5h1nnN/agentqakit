import type { TraceEvent } from "./types";

export interface TraceNode {
  event: TraceEvent;
  children: TraceNode[];
}

export function percent(value: number | null, digits = 0): string {
  return value === null ? "N/A" : `${(value * 100).toFixed(digits)}%`;
}

export function milliseconds(start: string, end: string): number {
  return Math.max(0, new Date(end).getTime() - new Date(start).getTime());
}

export function duration(value: number | null): string {
  if (value === null) return "N/A";
  return value < 1000 ? `${value.toFixed(0)} ms` : `${(value / 1000).toFixed(2)} s`;
}

export function shortId(value: string): string {
  return value.length <= 12 ? value : `${value.slice(0, 8)}…${value.slice(-4)}`;
}

export function buildTraceTree(events: TraceEvent[]): TraceNode[] {
  const nodes = new Map(events.map((event) => [event.span_id, { event, children: [] as TraceNode[] }]));
  const roots: TraceNode[] = [];
  for (const node of nodes.values()) {
    const parent = node.event.parent_span_id ? nodes.get(node.event.parent_span_id) : undefined;
    if (parent) parent.children.push(node);
    else roots.push(node);
  }
  const sort = (values: TraceNode[]) => {
    values.sort((a, b) => a.event.started_at.localeCompare(b.event.started_at));
    values.forEach((node) => sort(node.children));
  };
  sort(roots);
  return roots;
}
