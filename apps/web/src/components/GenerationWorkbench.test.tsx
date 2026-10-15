import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GenerationWorkbench } from "./GenerationWorkbench";

const hash = `sha256:${"a".repeat(64)}`;
const session = {
  session_id: "generation-1",
  workflow_id: "workflow-1",
  project_id: "demo",
  status: "PLAN_READY",
  idempotency_key: "request-1",
  input_hash: hash,
  goal: { categories: ["security"], risk_levels: ["high"], target_capabilities: ["answer"], target_policies: ["no-disclosure"], allowed_mutations: [], max_cases: 1, instructions: null, code_evidence_hashes: [] },
  evidence_mode: "MANIFEST_ONLY",
  code_evidence_hashes: [],
  max_output_tokens: 4096,
  max_total_tokens: 100,
  max_cost_usd: null,
  consumed_tokens: 10,
  consumed_cost_usd: null,
  plan_path: ".agentqa/plan.json",
  plan_checksum: hash,
  report_path: null,
  report_checksum: null,
  llm_calls: [{ provider: "deepseek", model: "deepseek-chat", status: "SUCCEEDED", token_usage: { total_tokens: 10 }, latency_ms: 12, evidence_hash: hash }],
  error: null,
  created_at: "2026-10-15T00:00:00Z",
  updated_at: "2026-10-15T00:00:01Z",
};

const workflow = {
  workflow_id: "workflow-1",
  project_id: "demo",
  name: "AI review",
  description: "",
  status: "READY" as const,
  revision: 3,
  idempotency_key: "workflow-request",
  input_hash: hash,
  input_snapshot: {},
  created_at: session.created_at,
  updated_at: session.updated_at,
  started_at: null,
  finished_at: null,
  error: null,
  steps: [],
};

const project = {
  project_id: "demo",
  name: "Demo",
  description: "",
  workspace_path: ".",
  default_spec_refs: { manifest: "manifest.yaml", generation_goal: "goal.yaml" },
  tags: [],
  recent_workflow_id: null,
  last_manifest_snapshot: null,
  created_at: session.created_at,
  updated_at: session.updated_at,
};

describe("GenerationWorkbench", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      const payload = path.endsWith("/generation-sessions")
        ? { sessions: [session] }
        : path === "/api/generation-sessions/generation-1"
          ? {
              session,
              plan: {
                generation_id: "generation-1",
                manifest_hash: hash,
                goal_hash: hash,
                items: [{ id: "security-refusal", title: "Refuse disclosure", category: "security", risk: "high", capabilities: ["answer"], policies: ["no-disclosure"], tools: [], mutation: null, rationale: "Declared policy coverage", source_refs: ["manifest:policy/no-disclosure"], provenance: ["declared"], requires_human: false }],
                questions: ["Should refusal wording be standardized?"],
                warnings: [],
              },
              candidates: [],
            }
          : { session: { ...session, status: "GENERATING" }, step: {} };
      return { ok: true, status: 200, json: async () => payload };
    }));
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows the separate plan phase and submits checksum confirmation", async () => {
    render(<GenerationWorkbench project={project} workflow={workflow} codeEvidence={[]} />);

    expect(await screen.findByText("Plan matrix")).toBeVisible();
    expect(screen.getByText("Should refusal wording be standardized?")).toBeVisible();
    expect(screen.getByText("unobservable")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Confirm plan & generate candidates" }));

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      "/api/generation-sessions/generation-1/candidates",
      expect.objectContaining({ method: "POST", body: expect.stringContaining(hash) }),
    ));
  });
});
