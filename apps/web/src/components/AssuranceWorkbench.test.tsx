import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, it, expect, vi } from "vitest";

import type { CodeEvidence, WorkflowRun } from "../types";
import { AssuranceWorkbench } from "./AssuranceWorkbench";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("analyzes a verified failure on a failed workflow without a preview click", async () => {
  const workflow: WorkflowRun = {
    workflow_id: "workflow-1", project_id: "demo", name: "Demo", description: "",
    status: "FAILED_SAFE", revision: 1, idempotency_key: "demo-1", input_hash: "hash",
    input_snapshot: {}, created_at: "2026-10-19T00:00:00Z", updated_at: "2026-10-19T00:00:00Z",
    started_at: null, finished_at: null, error: { code: "CHECKSUM", message: "error" }, steps: [],
  };
  const codeEvidence = [{ contentHash: `sha256:${"a".repeat(64)}`, path: "src/agent.py", startLine: 1, endLine: 4 }] as CodeEvidence[];
  const failure = {
    failure_id: "failure-1", workflow_id: "workflow-1", campaign_id: "campaign-1",
    run_id: "run-1", target_role: "defect", target_name: "agent", test_case_id: "case-1",
    trial_id: "trial-1", trace_id: "trace-1", risk: "high", evaluator_statuses: {},
    failed_evaluators: ["schema"], tools: [], mutation_ids: [], test_revision_hash: "hash",
    failure_bundle_ref: "failure.json", replay_cassette_ref: null, trace_status: "complete",
    analysis_status: "NOT_ANALYZED", analysis_count: 0, latest_analysis_id: null, latest_analysis_ref: null,
  };
  vi.stubGlobal("fetch", vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
    const path = String(input);
    const payload = path.includes("/failures/failure-1/analyses")
      ? { reused: true, report: null, step: null }
      : path.includes("/failures")
        ? { failures: [failure] }
        : path.includes("/analyses")
          ? { analyses: [] }
          : { runs: [{ run_id: "run-1", status: "COMPLETED" }], jobs: [] };
    return { ok: true, status: 200, json: async () => payload };
  }));

  render(<AssuranceWorkbench workflow={workflow} codeEvidence={codeEvidence} />);
  fireEvent.click(await screen.findByRole("button", { name: /Analyze failure with AI/ }));

  await waitFor(() => expect(fetch).toHaveBeenCalledWith(
    "/api/workflows/workflow-1/failures/failure-1/analyses",
    expect.objectContaining({
      method: "POST",
      body: expect.stringContaining(codeEvidence[0].contentHash),
    }),
  ));
  expect(fetch).not.toHaveBeenCalledWith(
    expect.stringContaining("evidence-preview"), expect.anything(),
  );
});
