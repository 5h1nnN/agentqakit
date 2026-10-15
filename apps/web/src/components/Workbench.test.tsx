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

describe("Workbench", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input);
        const payload = path === "/api/projects"
          ? { projects: [{ project_id: "demo", name: "Demo", description: "", created_at: workflow.created_at, updated_at: workflow.updated_at }] }
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

  it("shows persisted events and sends an explicit resume transition", async () => {
    render(<Workbench />);
    fireEvent.click(await screen.findByRole("button", { name: /Interrupted check/ }));

    expect(await screen.findByText("WORKFLOW_PROCESS_RESTART")).toBeVisible();
    expect(await screen.findByText("WORKFLOW CREATED")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      "/api/workflows/workflow-interrupted-1/resume",
      expect.objectContaining({ method: "POST" }),
    ));
  });
});
