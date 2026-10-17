import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Workbench } from "./Workbench";

const workflow = {
  workflow_id: "workflow-interrupted-1",
  project_id: "demo",
  name: "Interrupted check",
  description: "",
  status: "INTERRUPTED",
  revision: 2,
  created_at: "2026-10-15T00:00:00Z",
  updated_at: "2026-10-15T00:01:00Z",
  started_at: "2026-10-15T00:00:30Z",
  finished_at: null,
  error: { code: "WORKFLOW_PROCESS_RESTART", message: "Process restarted" },
  steps: [],
};

const project = {
  project_id: "demo",
  name: "Demo",
  description: "",
  workspace_path: ".",
  default_spec_refs: { manifest: "manifest.yaml", target: "target.yaml", tests: "tests" },
  tags: [],
  recent_workflow_id: null,
  last_manifest_snapshot: null,
  created_at: workflow.created_at,
  updated_at: workflow.updated_at,
};

const codeEvidence = {
  path: "src/agent.py",
  startLine: 1,
  endLine: 2,
  contentHash: `sha256:${"a".repeat(64)}`,
  reason: "Inspect agent behavior",
  content: "def answer():\n    return 'safe'",
};

describe("Workbench", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input);
        const payload = path === "/api/projects"
          ? { projects: [project] }
          : path === "/api/runs?project_id=demo"
            ? { runs: [], jobs: [] }
          : path === "/api/projects/demo"
            ? project
            : path === "/api/projects/demo/discover"
              ? { project, refs: project.default_spec_refs, ready: true, missing: [] }
            : path === "/api/projects/demo/preflight"
              ? { project_id: "demo", checked_at: workflow.updated_at, executable: true, requires_approval: true, checks: [{ check_id: "provider", category: "provider", status: "REVIEW", message: "Provider is configured.", path: null }], manifest_snapshot: null }
              : path === "/api/projects/demo/artifacts/import"
                ? { project_id: "demo", scanned: 1, imported: 1, unchanged: 0, rejected: 0, runs_registered: 0, issues: [] }
                : path === "/api/projects/demo/artifacts"
                  ? { artifacts: [{ artifact_id: "artifact-1", project_id: "demo", artifact_type: "GenerationReport", source: "REFERENCE", path: "benchmark/reference-results/demo/generation-report.json", schema_version: "agentqa.generation-report/v1alpha1", checksum: `sha256:${"b".repeat(64)}`, size_bytes: 200, created_at: workflow.created_at, workflow_id: null, run_id: null, availability: "AVAILABLE", metadata: {} }], total: 1 }
                  : path === "/api/projects/demo/code-evidence/discover"
                    ? { codeEvidence: [codeEvidence], count: 1 }
                      : path === "/api/projects/demo/code-evidence"
                        ? { codeEvidence: [codeEvidence] }
                        : path === "/api/projects/demo/evidence-preview"
                          ? { project_id: "demo", manifest_sections: ["capabilities"], code_evidence: [codeEvidence], evidence_types: ["manifest-summary", "reviewed-code-evidence"], excluded: ["API keys"], provider: "deepseek", model: "deepseek-chat", max_input_tokens: 12000, max_output_tokens: 4096, timeout_seconds: 60, max_cost_usd: null, evidence_hash: `sha256:${"c".repeat(64)}` }
          : path.startsWith("/api/workflows/workflow-interrupted-1/events")
            ? { events: [{ event_id: "event-1", workflow_id: workflow.workflow_id, revision: 1, event_type: "WORKFLOW_CREATED", occurred_at: workflow.created_at, message: "Workflow is ready.", step_id: null, data: {} }], revision: 1 }
            : path === "/api/workflows/workflow-interrupted-1/resume" && init?.method === "POST"
              ? { ...workflow, status: "READY", revision: 3, error: null }
              : { workflows: [workflow] };
        return { ok: true, status: 200, json: async () => payload };
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows the active evaluation and sends an explicit resume transition", async () => {
    render(<Workbench />);

    expect(await screen.findByText("WORKFLOW_PROCESS_RESTART")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Resume evaluation" }));

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      "/api/workflows/workflow-interrupted-1/resume",
      expect.objectContaining({ method: "POST" }),
    ));
  });

  it("auto-discovers project configuration and repository context", async () => {
    render(<Workbench />);

    fireEvent.click(await screen.findByRole("button", { name: "Scan project" }));
    expect(await screen.findByText("Ready")).toBeVisible();
    expect(screen.getByText(/resources were detected/)).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: /Test design/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Build AI context/ }));

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      "/api/projects/demo/code-evidence/discover",
      expect.objectContaining({ method: "POST" }),
    ));
    expect(await screen.findByText("src/agent.py")).toBeVisible();
  });
});
