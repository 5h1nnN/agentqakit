import { expect, test } from "@playwright/test";

import {
  completeAssurance,
  currentRecordRunIds,
  replayRecordedFailure,
  runFixedMutationCampaign,
} from "./real-deepseek-helpers";

const enabled = process.env.AGENTQA_REAL_DEEPSEEK === "1";
const projectId = process.env.AGENTQA_REAL_PROJECT_ID ?? "t22-real-deepseek";

test.skip(!enabled, "Real DeepSeek acceptance is explicit opt-in and never runs in default CI.");

test("real DeepSeek Web Golden Path writes a leak-scannable reference report", async ({ page }) => {
  test.setTimeout(15 * 60_000);
  await page.goto("/");
  await page.getByRole("button", { name: "New project" }).last().click();
  const projectForm = page.locator("form.quick-project");
  await projectForm.getByLabel("Project name").fill("Campus Ops T22 DeepSeek");
  await projectForm.getByLabel("Repository directory").fill("examples/showcase/campus_ops_agent");
  await projectForm.getByText("Optional details").click();
  await projectForm.getByLabel("Project ID").fill(projectId);
  await projectForm.getByRole("button", { name: "Create and scan" }).click();
  await expect(page.getByRole("heading", { name: "Campus Ops T22 DeepSeek" })).toBeVisible();
  await page.getByRole("button", { name: "Scan project" }).click();
  await expect(page.getByText("Ready", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: /Test design/ }).click();
  await page.getByRole("button", { name: "Build AI context" }).click();
  await expect(page.locator(".context-files article").first()).toBeVisible();

  await page.getByRole("button", { name: "New goal" }).click();
  await page.getByLabel("Goal name").fill("t22-live-acceptance");
  await page.getByLabel("Allowed mutations").fill("score-wrong-value,registration-false-duplicate,policy-indirect-injection");
  await page.getByLabel("Maximum candidates").fill("6");
  await page.getByLabel("Instructions").fill("Prioritize deterministic authorization, confirmation, retrieved-content trust, and mutation-recovery assertions.");
  await page.getByRole("button", { name: "Repository context" }).click();
  const evidenceChecks = page.locator(".evidence-options input[type=checkbox]");
  expect(await evidenceChecks.count()).toBeGreaterThan(0);
  await page.getByText("Usage limits").click();
  await page.getByLabel("Maximum response tokens").fill("8192");
  await page.getByRole("button", { name: "Generate test plan" }).click();
  const generate = page.getByRole("button", { name: "Confirm plan & generate candidates" });
  await expect(generate).toBeVisible({ timeout: 180_000 });
  await generate.click();
  await expect(page.getByRole("heading", { name: "Candidate review" })).toBeVisible({ timeout: 240_000 });
  const cards = page.locator(".candidate-card");
  const candidateCount = await cards.count();
  expect(candidateCount).toBeGreaterThan(0);
  for (let index = 0; index < candidateCount; index += 1) {
    await cards.nth(index).getByRole("button", { name: "Approve", exact: true }).click();
  }

  await page.getByRole("button", { name: /Execute/ }).click();
  const testSection = page.locator(".campaign-builder-grid > section").nth(0);
  const goldenLabels = testSection.locator(".campaign-options label");
  for (let index = 0; index < await goldenLabels.count(); index += 1) {
    const label = goldenLabels.nth(index);
    const keep = (await label.textContent())?.includes("score-other-student-denied") ?? false;
    const checkbox = label.locator("input");
    if (keep) await checkbox.check(); else await checkbox.uncheck();
  }
  const targetSection = page.locator(".campaign-builder-grid > section").nth(1);
  const targetLabels = targetSection.locator(".campaign-options label");
  for (let index = 0; index < await targetLabels.count(); index += 1) {
    const label = targetLabels.nth(index);
    const keep = /defect|fixed/.test((await label.textContent()) ?? "");
    const checkbox = label.locator("input");
    if (keep) await checkbox.check(); else await checkbox.uncheck();
  }
  await targetSection.getByText("Mutation campaign").locator("..").getByRole("checkbox").check();
  const runtime = page.locator(".campaign-builder-grid > section").nth(2);
  await runtime.getByLabel("Replay mode").selectOption("record");
  await runtime.getByText("Allow declared external side effects").locator("..").getByRole("checkbox").check();
  await page.getByRole("button", { name: "Freeze campaign config" }).click();
  await page.getByRole("button", { name: "Generate one-time confirmation" }).click();
  const phrase = (await page.locator(".confirmation-box code").textContent())?.trim() ?? "";
  await page.locator(".confirmation-box input").first().fill(phrase);
  await page.getByRole("button", { name: "Confirm exact config" }).click();
  await page.getByRole("button", { name: "Start campaign" }).click();
  await expect(page.locator(".campaign-review .review-heading .status")).toHaveText("COMPLETED", {
    timeout: 300_000,
  });

  const recordRuns = await currentRecordRunIds(page);
  await replayRecordedFailure(page);
  const mutationRunId = await runFixedMutationCampaign(page);
  await completeAssurance(page, projectId, { ...recordRuns, mutationRunId });
});
