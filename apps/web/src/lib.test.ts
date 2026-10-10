import { describe, expect, it } from "vitest";

import { buildTraceTree, duration, percent } from "./lib";
import type { TraceEvent } from "./types";

const event = (spanId: string, parentSpanId: string | null, startedAt: string): TraceEvent => ({
  trace_id: "trace",
  span_id: spanId,
  parent_span_id: parentSpanId,
  kind: "TOOL",
  name: spanId,
  started_at: startedAt,
  ended_at: startedAt,
  status: "OK",
  attributes: {},
  input: null,
  output: null,
});

describe("presentation helpers", () => {
  it("formats absent and observed metrics", () => {
    expect(percent(null)).toBe("N/A");
    expect(percent(0.875, 1)).toBe("87.5%");
    expect(duration(1250)).toBe("1.25 s");
  });

  it("builds and sorts a trace tree", () => {
    const tree = buildTraceTree([
      event("child-b", "root", "2026-10-10T00:00:02Z"),
      event("root", null, "2026-10-10T00:00:00Z"),
      event("child-a", "root", "2026-10-10T00:00:01Z"),
    ]);

    expect(tree).toHaveLength(1);
    expect(tree[0].event.span_id).toBe("root");
    expect(tree[0].children.map((child) => child.event.span_id)).toEqual(["child-a", "child-b"]);
  });
});
