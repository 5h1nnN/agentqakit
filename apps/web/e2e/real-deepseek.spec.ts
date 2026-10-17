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
  await page.getByRole("button", { name: "Workbench" }).click();

  const projectForm = page.locator("form.workbench-form").first();
  await projectForm.getByLabel("Project ID").fill(projectId);
  await projectForm.getByLabel("Display name").fill("Campus Ops T22 DeepSeek");
  await projectForm.getByLabel("Project directory").fill("examples/showcase/campus_ops_agent");
  await projectForm.getByText("Default specifications").click();
  await projectForm.getByLabel("Manifest").fill("agentqa/manifest-v2.yaml");
  await projectForm.getByLabel("Target", { exact: true }).fill("agentqa/targets/fixed.yaml");
  await projectForm.getByLabel("Defect target").fill("agentqa/targets/bug-authorization.yaml");
  await projectForm.getByLabel("Fixed target").fill("agentqa/targets/fixed.yaml");
  await projectForm.getByLabel("Tests").fill("agentqa/golden-tests/campus-ops.yaml");
  await projectForm.getByLabel("Mutations").fill("agentqa/mutations.yaml");
  await projectForm.getByLabel("Gate").fill("agentqa/regression-gate.yaml");
  await projectForm.getByLabel("Generation goal").fill("agentqa/generation-goal.yaml");
  await projectForm.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name: "Campus Ops T22 DeepSeek" })).toBeVisible();
  await page.getByRole("button", { name: "Run preflight" }).click();
  await expect(page.getByText("EXECUTABLE", { exact: true })).toBeVisible();

  const evidence = [
    ["src/campus_ops_agent/service.py", "43", "64", "Authorization order around private tools."],
    ["src/campus_ops_agent/service.py", "100", "132", "Bounded retry loop and tool response validation."],
    ["src/campus_ops_agent/service.py", "134", "144", "Confirmation-protected write call and idempotency key flow."],
    ["src/campus_ops_agent/service.py", "146", "163", "Retrieved-content trust boundary."],
    ["src/campus_ops_agent/policies.py", "28", "39", "Confirmation and required-parameter policy."],
    ["src/campus_ops_agent/policies.py", "42", "43", "Stable idempotency-key derivation."],
    ["src/campus_ops_agent/tools.py", "63", "77", "Write deduplication and duplicate defect branch."],
  ];
  const picker = page.locator("form.code-selection");
  for (const [file, start, end, reason] of evidence) {
    await picker.getByLabel("File").fill(file);
    await picker.getByLabel("Start line").fill(start);
    await picker.getByLabel("End line").fill(end);
    await picker.getByLabel("Selection reason").fill(reason);
    await picker.getByRole("button", { name: "Preview redacted excerpt" }).click();
    await page.getByRole("button", { name: "Add to CodeEvidence" }).click();
  }
  await expect(page.getByText("7/10 selected")).toBeVisible();

  const workflowForm = page.locator("form.workbench-form").nth(1);
  await workflowForm.getByLabel("Workflow name").fill("T22 DeepSeek Web Golden Path");
  await workflowForm.getByRole("button", { name: "Create workflow" }).click();
  await expect(page.getByRole("heading", { name: "Generation & review studio" })).toBeVisible();

  await page.getByRole("button", { name: "New goal" }).click();
  await page.getByLabel("Goal name").fill("t22-live-acceptance");
  await page.getByLabel("Allowed mutations").fill("score-wrong-value,registration-false-duplicate,policy-indirect-injection");
  await page.getByLabel("Maximum candidates").fill("6");
  await page.getByLabel("Instructions").fill("Prioritize deterministic authorization, confirmation, retrieved-content trust, and mutation-recovery assertions.");
  await page.getByRole("button", { name: "Manifest + code" }).click();
  const evidenceChecks = page.locator(".evidence-options input[type=checkbox]");
  await expect(evidenceChecks).toHaveCount(7);
  for (let index = 0; index < 7; index += 1) await evidenceChecks.nth(index).check();
  await page.getByLabel("Max output tokens / call").fill("8192");
  await page.getByRole("button", { name: "Generate plan" }).click();
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
