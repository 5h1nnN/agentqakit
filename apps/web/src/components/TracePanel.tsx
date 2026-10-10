import { buildTraceTree, duration, milliseconds, shortId, type TraceNode } from "../lib";
import type { TraceRecord } from "../types";
import { StatusBadge } from "./StatusBadge";

function Node({ node, depth = 0 }: { node: TraceNode; depth?: number }) {
  const event = node.event;
  return (
    <li>
      <details open={depth < 2}>
        <summary style={{ paddingLeft: `${depth * 14}px` }}>
          <span className="trace-kind">{event.kind}</span>
          <span>{event.name}</span>
          <span className="mono muted">{shortId(event.span_id)}</span>
          <span>{duration(milliseconds(event.started_at, event.ended_at))}</span>
          <StatusBadge status={event.status} />
        </summary>
        <div className="trace-detail">
          <details><summary>Attributes</summary><pre>{JSON.stringify(event.attributes, null, 2)}</pre></details>
          {event.input !== null && <details><summary>Input</summary><pre>{JSON.stringify(event.input, null, 2)}</pre></details>}
          {event.output !== null && <details><summary>Output</summary><pre>{JSON.stringify(event.output, null, 2)}</pre></details>}
        </div>
      </details>
      {node.children.length > 0 && <ul>{node.children.map((child) => <Node key={child.event.span_id} node={child} depth={depth + 1} />)}</ul>}
    </li>
  );
}

export function TracePanel({ trace }: { trace: TraceRecord | null }) {
  if (!trace) return null;
  if (trace.status === "missing") {
    return (
      <section className="panel observable-state">
        <StatusBadge status="missing">NOT OBSERVABLE</StatusBadge>
        <h3>Trace evidence is unavailable</h3>
        <p>{trace.diagnostics.join(" ") || "No target-internal spans were captured."}</p>
      </section>
    );
  }
  return (
    <section className="panel">
      <div className="panel__header">
        <div><p className="eyebrow">Execution evidence</p><h3>Trace tree</h3></div>
        <StatusBadge status={trace.status} />
      </div>
      {trace.diagnostics.length > 0 && <div className="notice notice--warn">{trace.diagnostics.join(" ")}</div>}
      <ul className="trace-tree">{buildTraceTree(trace.events).map((node) => <Node key={node.event.span_id} node={node} />)}</ul>
    </section>
  );
}
