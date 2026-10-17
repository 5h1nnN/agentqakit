import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { expect, type Page } from "@playwright/test";

function rowForRole(page: Page, role: string) {
  return page.locator(".campaign-matrix tbody tr").filter({
    has: page.getByText(role, { exact: true }),
  });
}

export async function currentRecordRunIds(
  page: Page,
): Promise<{ defectRunId: string; fixedRunId: string }> {
  const defectRunId = await rowForRole(page, "defect").locator("code").getAttribute("title");
  const fixedRunId = await rowForRole(page, "fixed").locator("code").getAttribute("title");
  expect(defectRunId).toBeTruthy();
  expect(fixedRunId).toBeTruthy();
  return { defectRunId: defectRunId ?? "", fixedRunId: fixedRunId ?? "" };
}

export async function replayRecordedFailure(
  page: Page,
  caseId = "score-other-student-denied",
): Promise<string> {
  await page.getByRole("button", { name: /Investigate/ }).click();
  await expect(page.locator(".failure-list button").first()).toBeVisible({ timeout: 60_000 });
  await page.locator(".failure-list button").first().click();
  const cassette = await page.getByLabel("Replay cassette path").inputValue();
  expect(cassette).toMatch(/\/cassettes\/.+\.jsonl$/);
  await page.getByRole("button", { name: /Execute/ }).click();

  const testSection = page.locator(".campaign-builder-grid > section").nth(0);
  const goldenLabels = testSection.locator(".campaign-options label");
  for (let index = 0; index < await goldenLabels.count(); index += 1) {
    const label = goldenLabels.nth(index);
    const checkbox = label.locator("input");
    if ((await label.textContent())?.includes(caseId)) {
      await checkbox.check();
    } else {
      await checkbox.uncheck();
    }
  }

  const targetSection = page.locator(".campaign-builder-grid > section").nth(1);
  const targetLabels = targetSection.locator(".campaign-options label");
  for (let index = 0; index < await targetLabels.count(); index += 1) {
    const label = targetLabels.nth(index);
    const checkbox = label.locator("input");
    if ((await label.locator("strong").textContent())?.trim() === "defect") {
      await checkbox.check();
    } else {
      await checkbox.uncheck();
    }
  }
  const mutation = targetSection.getByText("Mutation campaign").locator("..").getByRole("checkbox");
  if (await mutation.isChecked()) await mutation.uncheck();

  const runtime = page.locator(".campaign-builder-grid > section").nth(2);
  await runtime.getByLabel("Replay mode").selectOption("replay");
  await runtime.getByLabel("Cassette path").fill(cassette);
  const sideEffects = runtime
    .getByText("Allow declared external side effects")
    .locator("..")
    .getByRole("checkbox");
  if (!(await sideEffects.isChecked())) await sideEffects.check();

  await page.getByRole("button", { name: "Freeze campaign config" }).click();
  await page.getByRole("button", { name: "Generate one-time confirmation" }).click();
  const phrase = (await page.locator(".confirmation-box code").textContent())?.trim() ?? "";
  await page.locator(".confirmation-box input").first().fill(phrase);
  await page.getByRole("button", { name: "Confirm exact config" }).click();
  await page.getByRole("button", { name: "Start campaign" }).click();
  await expect(page.locator(".campaign-review .review-heading .status")).toHaveText("COMPLETED", {
    timeout: 180_000,
  });
  const replayRow = rowForRole(page, "defect");
  await expect(replayRow.locator("td").nth(8)).toHaveText(/^[1-9]\d*\/0\/0$/);
  return (await replayRow.locator("code").getAttribute("title")) ?? "";
}

export async function runFixedMutationCampaign(page: Page): Promise<string> {
  const testSection = page.locator(".campaign-builder-grid > section").nth(0);
  const goldenLabels = testSection.locator(".campaign-options label");
  for (let index = 0; index < await goldenLabels.count(); index += 1) {
    const label = goldenLabels.nth(index);
    const checkbox = label.locator("input");
    if ((await label.textContent())?.includes("course-ai101")) {
      await checkbox.check();
    } else {
      await checkbox.uncheck();
    }
  }

  const targetSection = page.locator(".campaign-builder-grid > section").nth(1);
  const targetLabels = targetSection.locator(".campaign-options label");
  for (let index = 0; index < await targetLabels.count(); index += 1) {
    const label = targetLabels.nth(index);
    const checkbox = label.locator("input");
    if ((await label.locator("strong").textContent())?.trim() === "fixed") {
      await checkbox.check();
    } else {
      await checkbox.uncheck();
    }
  }
  const mutation = targetSection.getByText("Mutation campaign").locator("..").getByRole("checkbox");
  if (!(await mutation.isChecked())) await mutation.check();

  const runtime = page.locator(".campaign-builder-grid > section").nth(2);
  await runtime.getByLabel("Replay mode").selectOption("off");
  const sideEffects = runtime
    .getByText("Allow declared external side effects")
    .locator("..")
    .getByRole("checkbox");
  if (!(await sideEffects.isChecked())) await sideEffects.check();

  await page.getByRole("button", { name: "Freeze campaign config" }).click();
  await page.getByRole("button", { name: "Generate one-time confirmation" }).click();
  const phrase = (await page.locator(".confirmation-box code").textContent())?.trim() ?? "";
  await page.locator(".confirmation-box input").first().fill(phrase);
  await page.getByRole("button", { name: "Confirm exact config" }).click();
  await page.getByRole("button", { name: "Start campaign" }).click();
  await expect(page.locator(".campaign-review .review-heading .status")).toHaveText("COMPLETED", {
    timeout: 240_000,
  });
  const fixedRow = rowForRole(page, "fixed");
  await expect(fixedRow.locator("td").nth(9)).toHaveText(/^[1-9]\d* ·/);
  return (await fixedRow.locator("code").getAttribute("title")) ?? "";
}

export async function completeAssurance(
  page: Page,
  projectId: string,
  options: {
    startAtComparison?: boolean;
    defectRunId?: string;
    fixedRunId?: string;
    mutationRunId?: string;
  } = {},
): Promise<void> {
  const defectRun =
    options.defectRunId ??
    (await rowForRole(page, "defect").locator("code").getAttribute("title"));
  const fixedRun =
    options.fixedRunId ??
    (await rowForRole(page, "fixed").locator("code").getAttribute("title"));
  expect(defectRun).toBeTruthy();
  expect(fixedRun).toBeTruthy();

  if (!options.startAtComparison) {
    await page.getByRole("button", { name: /Investigate/ }).click();
    await expect(page.locator(".failure-list button").first()).toBeVisible({ timeout: 60_000 });
    await page.getByRole("button", { name: "Review AI input" }).click();
    await page.getByRole("button", { name: "Analyze failure with AI" }).click();
    await expect(page.getByText(/Root-cause candidates/)).toBeVisible({
      timeout: 240_000,
    });
    await page.getByRole("button", { name: "Create bounded Handoff Bundle" }).click();
    await expect(page.getByLabel("MCP resource URI")).toHaveValue(/agentqa:\/\/workflows\//);
    await page.getByRole("button", { name: "Check repaired version" }).click();
    await page.getByRole("button", { name: /Overview/ }).click();
    const resume = page.getByRole("button", { name: "Resume evaluation" });
    await expect(resume).toBeEnabled({ timeout: 10_000 });
    await resume.click();
  }

  await page.getByRole("button", { name: /Release/ }).click();
  await page.locator(".assurance-workbench > .panel__header").getByRole("button", { name: "Refresh" }).click();
  const compareSection = page.locator(".assurance-section").filter({
    has: page.getByRole("heading", { name: "Compare versions and apply a release gate" }),
  });
  const baseline = compareSection.locator("select").nth(0);
  const candidate = compareSection.locator("select").nth(1);
  await expect.poll(
    async () => baseline.locator("option").evaluateAll((items) => items.length),
    { timeout: 15_000 },
  ).toBeGreaterThan(1);
  const baselineValues = await baseline.locator("option").evaluateAll((items) => items.map((item) => (item as HTMLOptionElement).value));
  const candidateValues = await candidate.locator("option").evaluateAll((items) => items.map((item) => (item as HTMLOptionElement).value));
  expect(baselineValues).toContain(defectRun);
  expect(candidateValues).toContain(fixedRun);
  await baseline.selectOption({ index: baselineValues.indexOf(defectRun ?? "") });
  await candidate.selectOption({ index: candidateValues.indexOf(fixedRun ?? "") });
  await page.getByRole("button", { name: "Compare runs" }).click();
  await expect(page.getByText("Fixed").last()).toBeVisible();
  await page.getByLabel("Release policy").fill("agentqa/regression-gate.yaml");
  await page.getByRole("button", { name: "Make release decision" }).click();
  await expect(page.getByText(/meets the release policy/)).toBeVisible();
  await compareSection.locator("select").nth(2).selectOption("mutation");
  if (options.mutationRunId) {
    const mutationCandidateValues = await candidate
      .locator("option")
      .evaluateAll((items) => items.map((item) => (item as HTMLOptionElement).value));
    expect(mutationCandidateValues).toContain(options.mutationRunId);
    await candidate.selectOption({
      index: mutationCandidateValues.indexOf(options.mutationRunId),
    });
  }
  await page.getByLabel("Release policy").fill("agentqa/mutation-gate.yaml");
  await page.getByRole("button", { name: "Make release decision" }).click();
  await expect(page.getByRole("heading", { name: "mutation gate" })).toBeVisible();
  await expect(page.getByText(/meets the release policy/)).toBeVisible();
  await page.getByRole("button", { name: "Generate report" }).click();

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download JSON" }).click();
  const download = await downloadPromise;
  const output = path.resolve(process.cwd(), "../../benchmark/reference-results/t22-deepseek-web-20261015");
  await mkdir(output, { recursive: true });
  const reportPath = path.join(output, "workflow-report.json");
  await download.saveAs(reportPath);
  const bytes = await readFile(reportPath);
  const checksum = `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
  await writeFile(`${reportPath}.sha256`, `${checksum}\n`, "utf8");
  await writeFile(
    path.join(output, "acceptance.json"),
    `${JSON.stringify({
      schema_version: "agentqa.t22-acceptance/v1alpha1",
      completed_at: new Date().toISOString(),
      channel: "web",
      provider: "deepseek",
      project_id: projectId,
      workflow_id: report.workflow_id,
      identity: report.identity,
      generation: report.generation,
      run_ids: {
        defect: defectRun,
        fixed: fixedRun,
        mutation: options.mutationRunId ?? null,
      },
      gates: report.gates,
      workflow_report_checksum: checksum,
    }, null, 2)}\n`,
    "utf8",
  );
}
