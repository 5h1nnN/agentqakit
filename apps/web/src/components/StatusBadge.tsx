import type { ReactNode } from "react";

export function StatusBadge({ status, children }: { status: string; children?: ReactNode }) {
  const tone = ["COMPLETED", "PASS", "OK", "complete", "STABLE", "FIXED"].includes(status)
    ? "good"
    : ["QUEUED", "RUNNING", "partial", "NEEDS_REVIEW", "NEW", "REMOVED"].includes(status)
      ? "warn"
      : ["FAILED", "FAIL", "ERROR", "TIMEOUT", "missing", "NEW_REGRESSION"].includes(status)
        ? "bad"
        : "neutral";
  return <span className={`status status--${tone}`}>{children ?? status.replaceAll("_", " ")}</span>;
}
