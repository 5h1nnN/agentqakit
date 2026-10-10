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
              workspaceRoot: "/workspace",
              judge: { configured: true, provider: "deepseek", model: "deepseek-test" },
            }
          : { runs: [], jobs: [] },
      })),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders the empty ledger and opens the quality workspace", async () => {
    render(<App />);

    expect(await screen.findByRole("heading", { name: "No runs yet" })).toBeVisible();
    expect(screen.getByRole("checkbox", { name: /DeepSeek AI judge/ })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Compare & gate" }));

    expect(screen.getByRole("heading", { name: "Comparison & gate" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Evaluate gate" })).toBeDisabled();
  });
});
