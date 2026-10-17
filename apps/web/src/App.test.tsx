import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

describe("App", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(async (input: RequestInfo | URL) => ({
        ok: true,
        status: 200,
        json: async () => String(input).includes("/api/health")
          ? {
              status: "ok",
              storage: "SQLiteResultStore",
              workflowStorage: "SQLiteWorkflowStore",
              workspaceRoot: "/workspace",
              judge: { configured: true, provider: "deepseek", model: "deepseek-test" },
            }
          : String(input).includes("/api/projects")
            ? { projects: [] }
            : String(input).includes("/api/workflows")
              ? { workflows: [] }
          : String(input).includes("/api/ai/evidence")
            ? { generations: [], analyses: [] }
            : { runs: [], jobs: [] },
      })),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("opens the unified workspace and AI activity", async () => {
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Test an agent from idea to release." })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Runs" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "AI activity" }));
    expect(await screen.findByRole("heading", { name: "AI test generation" })).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Workspace" }));
    expect(await screen.findByRole("heading", { name: "Test an agent from idea to release." })).toBeVisible();
  });
});
