import { expect, test } from "@playwright/test";

import {
  completeAssurance,
  currentRecordRunIds,
  replayRecordedFailure,
  runFixedMutationCampaign,
} from "./real-deepseek-helpers";

const projectId = process.env.AGENTQA_REAL_RESUME_PROJECT_ID;
const workflowId = process.env.AGENTQA_REAL_RESUME_WORKFLOW_ID;
const resumeStage = process.env.AGENTQA_REAL_RESUME_STAGE;

test.skip(!projectId || !workflowId, "A persisted real acceptance workflow is required.");

test("resume a completed real campaign through replay, assurance gates, and report", async ({ page }) => {
  test.setTimeout(10 * 60_000);
  await page.goto("/");
  await page.getByRole("button", { name: "Workbench" }).click();
  const workflowForm = page.locator("form.workbench-form").nth(1);
  await workflowForm.getByLabel("Project").selectOption(projectId ?? "");
  await expect(page.getByRole("heading", { name: "Campus Ops T22 DeepSeek" })).toBeVisible();
  await page.getByRole("button", { name: "Refresh catalog" }).click();
  const workflow = page.locator(".workflow-item").filter({ hasText: (workflowId ?? "").slice(0, 9) });
  await workflow.click();
  await expect(page.locator(".campaign-review .review-heading .status")).toHaveText("COMPLETED");
  const configuredRuns = {
    defectRunId: process.env.AGENTQA_REAL_DEFECT_RUN_ID ?? "",
    fixedRunId: process.env.AGENTQA_REAL_FIXED_RUN_ID ?? "",
  };
  const recordRuns =
    configuredRuns.defectRunId && configuredRuns.fixedRunId
      ? configuredRuns
      : await currentRecordRunIds(page);
  if (resumeStage === "replay") await replayRecordedFailure(page);
  const mutationRunId =
    resumeStage === "replay" || resumeStage === "mutation"
      ? await runFixedMutationCampaign(page)
      : process.env.AGENTQA_REAL_MUTATION_RUN_ID;
  await completeAssurance(page, projectId ?? "", {
    startAtComparison:
      resumeStage === "compare" || resumeStage === "replay" || resumeStage === "mutation",
    mutationRunId,
    ...recordRuns,
  });
});
