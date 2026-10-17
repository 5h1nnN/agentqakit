import { expect, test, type Page, type Route } from "@playwright/test";

import { replayRecordedFailure } from "./real-deepseek-helpers";

const now = "2026-10-15T00:00:00Z";
const hash = `sha256:${"1".repeat(64)}`;
const codeEvidence = Array.from({ length: 7 }, (_, index) => ({
  path: `src/campus_ops/part_${index + 1}.py`,
  startLine: 1,
  endLine: 20,
  contentHash: `sha256:${String(index + 2).repeat(64).slice(0, 64)}`,
  reason: `Golden Path evidence ${index + 1}`,
  content: `def evidence_${index + 1}(): return true`,
}));

const project = {
  project_id: "campus-ops",
  name: "Campus Ops Agent",
  description: "T22 browser acceptance project",
  workspace_path: "examples/showcase/campus_ops_agent",
  default_spec_refs: {
    manifest: "examples/showcase/campus_ops_agent/agentqa/manifest-v2.yaml",
    tests: "examples/showcase/campus_ops_agent/agentqa/golden-tests/campus-ops.yaml",
    mutations: "examples/showcase/campus_ops_agent/agentqa/mutations.yaml",
    gate: "examples/showcase/campus_ops_agent/agentqa/regression-gate.yaml",
    defect_target: "examples/showcase/campus_ops_agent/agentqa/targets/bug-authorization.yaml",
    fixed_target: "examples/showcase/campus_ops_agent/agentqa/targets/fixed.yaml",
    generation_goal: "examples/showcase/campus_ops_agent/agentqa/generation-goal-code.yaml",
  },
  tags: ["t22", "golden-path"],
  recent_workflow_id: "workflow-t22",
  last_manifest_snapshot: {},
  created_at: now,
  updated_at: now,
};

const workflow = {
  workflow_id: "workflow-t22",
  project_id: "campus-ops",
  name: "T22 Web Golden Path",
  description: "Mock Provider no-network browser acceptance",
  status: "READY",
  revision: 7,
  idempotency_key: "t22-golden-path",
  input_hash: hash,
  input_snapshot: {},
  created_at: now,
  updated_at: now,
  started_at: now,
  finished_at: null,
  error: null,
  steps: [],
};

const runSummary = (passed: boolean) => ({
  total: 1,
  passed: passed ? 1 : 0,
  failed: passed ? 0 : 1,
  incomplete: 0,
  pass_rate: passed ? 1 : 0,
  inconsistent_cases: 0,
  cases: [],
});

const runs = [
  { project_id: "campus-ops", run_id: "run-fixed", status: "COMPLETED", exit_code: 0, started_at: now, ended_at: now, summary: runSummary(true), mutation_score: 1, artifact_path: "artifacts/runs/run-fixed", finding_count: 0 },
  { project_id: "campus-ops", run_id: "run-defect", status: "COMPLETED", exit_code: 1, started_at: now, ended_at: now, summary: runSummary(false), mutation_score: 0, artifact_path: "artifacts/runs/run-defect", finding_count: 1 },
];

const llmCall = { provider: "mock", model: "deterministic-v1", status: "SUCCEEDED", token_usage: { total_tokens: 120 }, latency_ms: 1, evidence_hash: hash };
const session = {
  session_id: "session-t22", workflow_id: "workflow-t22", project_id: "campus-ops",
  status: "CANDIDATES_READY", idempotency_key: "plan-t22", input_hash: hash,
  goal: { categories: ["security"], risk_levels: ["high"], target_capabilities: [], target_policies: [], allowed_mutations: [], max_cases: 1, instructions: null, code_evidence_hashes: codeEvidence.map((item) => item.contentHash) },
  evidence_mode: "MANIFEST_AND_CODE", code_evidence_hashes: codeEvidence.map((item) => item.contentHash),
  max_output_tokens: 4096, max_total_tokens: null, max_cost_usd: null, consumed_tokens: 120, consumed_cost_usd: null,
  plan_path: ".agentqa/artifacts/plan.json", plan_checksum: hash, report_path: ".agentqa/artifacts/generation-report.json", report_checksum: hash,
  llm_calls: [llmCall], error: null, created_at: now, updated_at: now,
};
const candidate = {
  case_id: "authorization-boundary", risk: "high", review_status: "REVIEW_REQUIRED",
  rationale: "Verify authorization policy under a controlled defect.", assertion_basis: ["policy:authorization"],
  coverage_item_ids: ["security-1"], source_refs: ["manifest:policies", `code:${codeEvidence[0].contentHash}`],
  original_spec_hash: hash, current_spec_hash: hash,
  spec: { apiVersion: "agentqa.dev/v1alpha1", kind: "AgentTest", metadata: { name: "authorization-boundary" }, scenario: { input: { action: "enroll" } }, expect: { output: { requireNonEmpty: true } }, evaluators: [{ type: "schema" }] },
  latest_revision: null, decision: null,
  diagnostics: [{ code: "VALID", status: "PASS", message: "Strict schema validation passed." }], similarity_hints: [], campaign_eligible: false,
};

const failure = {
  failure_id: "failure-t22", workflow_id: "workflow-t22", campaign_id: "campaign-t22", run_id: "run-defect",
  target_role: "defect", target_name: "bug-authorization", test_case_id: "authorization-boundary", trial_id: "trial-defect", trace_id: "trace-defect",
  risk: "high", evaluator_statuses: { schema: "FAIL" }, failed_evaluators: ["schema"], tools: ["enroll_student"], mutation_ids: ["deny-authorization"],
  test_revision_hash: hash, failure_bundle_ref: "artifacts/runs/run-defect/failures/failure-trial-defect.zip", trace_status: "complete",
  replay_cassette_ref: "artifacts/runs/run-defect/cassettes/trial-defect.jsonl",
  analysis_status: "ANALYZED", analysis_count: 1, latest_analysis_id: "analysis-t22", latest_analysis_ref: ".agentqa/artifacts/analysis-t22.json",
};
const analysis = {
  analysis_id: "analysis-t22", generated_at: now, run_id: "run-defect", test_case_id: "authorization-boundary", trial_id: "trial-defect", trace_id: "trace-defect", original_passed: false, evidence_hash: hash,
  evidence_index: [], observed_facts: [{ statement: "The defect bypassed authorization.", evidence_ids: ["evaluation-01"] }],
  earliest_divergence: { description: "Authorization guard returned allow.", evidence_ids: ["trace-event-01"], span_id: "span-1" },
  root_causes: [{ id: "authorization-guard", hypothesis: "The authorization predicate is inverted.", confidence: 0.91, evidence_ids: ["trace-event-01"], counter_evidence_ids: [], reasoning: "The first divergent tool decision admits a forbidden caller." }],
  missing_evidence: [], regression_tests: [{ title: "Reject unauthorized enrollment", rationale: "Protect the observed boundary.", expected_behavior: "Return a denial without calling the write tool.", evidence_ids: ["evaluation-01"] }], repair_directions: ["Inspect the authorization predicate."], llm_call: llmCall,
};

const campaign = {
  campaign_id: "campaign-t22", workflow_id: "workflow-t22", project_id: "campus-ops", idempotency_key: "campaign-t22", config_hash: hash,
  config: { replay_mode: "record", replay_cassette: null, judge_enabled: true, trials: 1, concurrency: 1, max_total_tokens: null, max_total_cost_usd: null, max_duration_seconds: 120, allow_side_effects: false },
  risk: { case_count: 1, target_count: 2, mutation_count: 1, max_trials: 2, high_risk_cases: ["authorization-boundary"], side_effect_tools: [], external_side_effects: [], irreversible_side_effects: [], requires_confirmation: true, reasons: ["high risk generated case"] },
  status: "COMPLETED", approval_id: "approval-t22", workflow_step_id: "step-campaign", completed_trials: 2, total_trials: 2, created_at: now, updated_at: now, started_at: now, finished_at: now, error: null,
  subruns: [
    { subrun_id: "sub-defect", target: { role: "defect", path: project.default_spec_refs.defect_target, name: "bug-authorization", spec_hash: hash }, run_id: "run-defect", prior_run_ids: [], attempt: 1, status: "COMPLETED", planned_trials: 1, completed_trials: 1, current_case_id: null, replay_hits: 0, replay_misses: 0, live_fallbacks: 1, judge_evaluations: 1, mutation_applications: 1, mutation_killed: 1, mutation_survived: 0, mutation_not_reached: 0, mutation_invalid: 0, summary: runSummary(false), budget: { consumed_total_tokens: 20, consumed_cost_usd: 0, exhausted: false }, artifact_path: "artifacts/runs/run-defect", started_at: now, finished_at: now, error: null },
    { subrun_id: "sub-fixed", target: { role: "fixed", path: project.default_spec_refs.fixed_target, name: "fixed", spec_hash: hash }, run_id: "run-fixed", prior_run_ids: [], attempt: 1, status: "COMPLETED", planned_trials: 1, completed_trials: 1, current_case_id: null, replay_hits: 1, replay_misses: 0, live_fallbacks: 0, judge_evaluations: 1, mutation_applications: 1, mutation_killed: 1, mutation_survived: 0, mutation_not_reached: 0, mutation_invalid: 0, summary: runSummary(true), budget: { consumed_total_tokens: 20, consumed_cost_usd: 0, exhausted: false }, artifact_path: "artifacts/runs/run-fixed", started_at: now, finished_at: now, error: null },
  ],
};

const replayCampaign = {
  ...campaign,
  campaign_id: "campaign-replay-t22",
  idempotency_key: "campaign-replay-t22",
  config: { ...campaign.config, replay_mode: "replay", replay_cassette: failure.replay_cassette_ref, judge_enabled: false },
  risk: { ...campaign.risk, target_count: 1, mutation_count: 0, max_trials: 1, reasons: ["high risk generated case"] },
  status: "AWAITING_CONFIRMATION",
  approval_id: null,
  completed_trials: 0,
  total_trials: 1,
  started_at: null,
  finished_at: null,
  subruns: [{
    ...campaign.subruns[0],
    subrun_id: "sub-replay-defect",
    run_id: "run-replay-defect",
    status: "PENDING",
    completed_trials: 0,
    replay_hits: 0,
    live_fallbacks: 0,
    summary: null,
    artifact_path: null,
    started_at: null,
    finished_at: null,
  }],
};

const comparison = {
  baseline_run_id: "run-defect", candidate_run_id: "run-fixed", candidate_effective_pass_rate: 1,
  counts: { new_regressions: 0, fixed: 1, persistent_failures: 0, stable: 0, new: 0, removed: 0 },
  baseline: { pass_rate: 0, mutation_score: 0 }, candidate: { pass_rate: 1, mutation_score: 1 },
  deltas: { pass_rate: 1, mutation_score: 1, high_risk_failures: -1, p95_latency_ms: { baseline: null, candidate: null, absolute_change: null, relative_increase: null, status: "unobservable" }, average_tokens: { baseline: 20, candidate: 20, absolute_change: 0, relative_increase: 0, status: "available" }, average_cost_usd: { baseline: null, candidate: null, absolute_change: null, relative_increase: null, status: "unobservable" } },
  quality_cost: { status: "QUALITY_IMPROVED_COST_NEUTRAL", quality_preserved: true, cost_improved: null, token_improved: false, latency_improved: null, reasons: ["Pass rate improved; cost remains unobservable."] },
  cases: [{ test_case_id: "authorization-boundary", classification: "FIXED", baseline_passed: false, candidate_passed: true, risk: "high" }],
};

async function mockApi(page: Page, calls: string[]) {
  let campaigns = [campaign];
  await page.route("**/api/**", async (route: Route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();
    const json = (value: unknown, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(value) });
    calls.push(`${method} ${path}`);

    if (path === "/api/health") return json({ status: "ok", storage: "Mock", workflowStorage: "Mock", workspaceRoot: "/workspace", judge: { configured: true, provider: "mock", model: "deterministic-v1" } });
    if (path === "/api/projects" && method === "GET") return json({ projects: [project] });
    if (path === "/api/projects/campus-ops" && method === "GET") return json(project);
    if (path === "/api/projects/campus-ops/preflight") return json({ project_id: "campus-ops", checked_at: now, executable: true, requires_approval: false, checks: [{ check_id: "manifest", category: "schema", status: "PASS", message: "Manifest is valid.", path: project.default_spec_refs.manifest }], manifest_snapshot: {} });
    if (path === "/api/projects/campus-ops/artifacts") return json({ artifacts: [], total: 0 });
    if (path === "/api/projects/campus-ops/code-evidence") return json({ codeEvidence });
    if (path === "/api/projects/campus-ops/evidence-preview") return json({ project_id: "campus-ops", manifest_sections: ["purpose", "capabilities", "tools", "policies"], code_evidence: codeEvidence, evidence_types: ["manifest", "code"], excluded: ["unselected source files", "secrets"], provider: "mock", model: "deterministic-v1", max_input_tokens: 32000, max_output_tokens: 4096, timeout_seconds: 60, max_cost_usd: null, evidence_hash: hash });
    if (path === "/api/workflows" && method === "GET") return json({ workflows: [workflow] });
    if (path === "/api/workflows/workflow-t22" && method === "GET") return json(workflow);
    if (path === "/api/workflows/workflow-t22/events") return json({ events: [], revision: 7 });
    if (path === "/api/workflows/workflow-t22/generation-sessions") return json({ sessions: [session] });
    if (path === "/api/generation-sessions/session-t22") return json({ session, plan: { generation_id: "gen-t22", manifest_hash: hash, goal_hash: hash, items: [{ id: "security-1", title: "Authorization boundary", category: "security", risk: "high", capabilities: [], policies: ["authorization"], tools: ["enroll_student"], mutation: "deny-authorization", rationale: "Cover the privileged write boundary.", source_refs: ["manifest:policies"], provenance: [], requires_human: true }], questions: [], warnings: [] }, candidates: [candidate] });
    if (path.includes("/candidates/") && path.endsWith("/review")) return json({ approval_id: "approval-candidate" });
    if (path === "/api/workflows/workflow-t22/campaign-inputs") return json({ goldenTestsPath: project.default_spec_refs.tests, golden: [{ caseId: "authorization-boundary", risk: "high", specHash: hash, sourceRef: project.default_spec_refs.tests }], generated: [{ sessionId: "session-t22", caseId: "authorization-boundary", risk: "high", specHash: hash, campaignEligible: true, reviewStatus: "APPROVED" }], targets: [{ role: "defect", path: project.default_spec_refs.defect_target }, { role: "fixed", path: project.default_spec_refs.fixed_target }], manifestPath: project.default_spec_refs.manifest, mutationsPath: project.default_spec_refs.mutations, judge: { available: true, provider: "mock", model: "deterministic-v1" } });
    if (path === "/api/workflows/workflow-t22/campaigns" && method === "GET") return json({ campaigns });
    if (path === "/api/workflows/workflow-t22/campaigns" && method === "POST") {
      campaigns = [replayCampaign, campaign];
      return json(replayCampaign, 201);
    }
    if (path === "/api/campaigns/campaign-replay-t22/challenge") return json({ challenge: { challenge_id: "challenge-t22", config_hash: hash, expires_at: now }, phrase: "CONFIRM T22 REPLAY", risk: replayCampaign.risk }, 201);
    if (path === "/api/campaigns/campaign-replay-t22/confirm") {
      const confirmed = { ...replayCampaign, status: "DRAFT", approval_id: "approval-replay-t22" };
      campaigns = [confirmed, campaign];
      return json(confirmed);
    }
    if (path === "/api/campaigns/campaign-replay-t22/start") {
      const completed = {
        ...replayCampaign,
        status: "COMPLETED",
        approval_id: "approval-replay-t22",
        completed_trials: 1,
        started_at: now,
        finished_at: now,
        subruns: [{ ...replayCampaign.subruns[0], status: "COMPLETED", completed_trials: 1, replay_hits: 1, summary: runSummary(false), artifact_path: "artifacts/runs/run-replay-defect", started_at: now, finished_at: now }],
      };
      campaigns = [completed, campaign];
      return json({ campaign: completed, step: {} });
    }
    if (path === "/api/workflows/workflow-t22/failures/failure-t22/evidence-preview") return json({ failure, evidence_hash: hash, bundle: { trace_status: "complete", items: [{ evidence_id: "trace-event-01", kind: "trace", source_ref: "trace-defect", content_hash: hash, summary: "Authorization guard decision", truncated: false, data: { allowed: true } }], missing_evidence: [], blocking_gaps: [] }, reusable_analyses: [analysis] });
    if (path === "/api/workflows/workflow-t22/failures/failure-t22/analyses") return json({ reused: true, report: analysis, step: null });
    if (path === "/api/workflows/workflow-t22/failures") return json({ failures: [failure] });
    if (path === "/api/workflows/workflow-t22/analyses") return json({ analyses: [analysis] });
    if (path === "/api/workflows/workflow-t22/handoffs") return json({ handoff: { handoff_id: "handoff-t22", created_at: now, workflow_id: "workflow-t22", project_id: "campus-ops", handoff_resource_uri: "agentqa://workflows/workflow-t22/handoffs/handoff-t22", failure_resource_uri: "agentqa://runs/run-defect/failures/trial-defect", analysis_resource_uri: "agentqa://analyses/analysis-t22", baseline_target_hash: hash, baseline_manifest_hash: hash, baseline_source_snapshot_hash: hash, baseline_runtime_hash: hash, suggested_regression_tests: ["Reject unauthorized enrollment"], missing_evidence: [], developer_task: "Fix the cited authorization defect and return a new target hash." }, artifact: { artifact_id: "handoff", artifact_type: "HandoffBundle", uri: ".agentqa/artifacts/handoff.json", schema_version: "agentqa.handoff-bundle/v1alpha1", checksum: hash, source: "workflow", verification_status: "VERIFIED" } }, 201);
    if (path === "/api/workflows/workflow-t22/repair-preflight") return json({ report: { project_id: "campus-ops", checked_at: now, executable: true, requires_approval: false, checks: [], manifest_snapshot: {} }, changes: { manifest: false, source: true, runtime: true } });
    if (path === "/api/workflows/workflow-t22/comparisons") return json({ report: comparison, step: {} });
    if (path === "/api/workflows/workflow-t22/gates") {
      const kind = (request.postDataJSON() as { kind?: string } | null)?.kind ?? "regression";
      return json({ gate: { gate_id: `gate-${kind}-t22`, kind, gate_path: kind === "mutation" ? "agentqa/mutation-gate.yaml" : project.default_spec_refs.gate, gate_spec_hash: hash, baseline_run_id: kind === "regression" ? "run-defect" : null, candidate_run_id: "run-fixed", result: { passed: true, exit_code: 0, rules: [{ name: kind === "mutation" ? "minimum-mutation-score" : "no-regression", passed: true, actual: kind === "mutation" ? 1 : 0, expected: kind === "mutation" ? 0.8 : 0, message: `${kind} gate passed.` }] } }, step: {} });
    }
    if (path === "/api/workflows/workflow-t22/reports") return json({ report: { schema_version: "agentqa.workflow-report/v1alpha1", report_id: "report-t22", generated_at: now, workflow_id: "workflow-t22", project: {}, identity: {}, provider: {}, generation: {}, test_results: {}, analyses: { total: 1 }, comparisons: [], gates: [], resources: {}, approvals: { count: 2 }, missing_evidence: [], authoritative_artifacts: [{ artifact_id: "run-fixed", artifact_type: "Run", uri: "artifacts/runs/run-fixed", schema_version: "agentqa.run/v1alpha1", checksum: hash, source: "workflow", verification_status: "VERIFIED" }] }, step: {}, artifact: { uri: ".agentqa/artifacts/workflow-report.json" } });
    if (path === "/api/runs") return json({ runs, jobs: [] });
    return json({});
  });
}

test("Campus Ops Golden Path is operable entirely from the Web workbench", async ({ page }) => {
  const calls: string[] = [];
  await mockApi(page, calls);
  await page.goto("/");
  await page.getByRole("button", { name: "Workbench" }).click();
  await expect(page.getByRole("heading", { name: "Assurance workbench" })).toBeVisible();

  await page.locator(".workflow-item", { hasText: "T22 Web Golden Path" }).click();
  await expect(page.getByRole("heading", { name: "Generation & review studio" })).toBeVisible();

  await page.getByRole("button", { name: "Run preflight" }).click();
  await expect(page.getByText("EXECUTABLE", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Refresh catalog" }).click();
  await expect(page.getByText("7/10 selected")).toBeVisible();

  await expect(page.getByText("mock / deterministic-v1").first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Candidate review" })).toBeVisible();
  await page.locator(".candidate-card").getByRole("button", { name: "Approve" }).click();
  await expect.poll(() => calls.some((value) => value.includes("/review"))).toBeTruthy();

  await expect(page.getByRole("heading", { name: "Campaign builder" })).toBeVisible();
  await expect(page.locator(".campaign-risk-grid").getByText("record", { exact: true })).toBeVisible();
  await expect(page.getByText("2 / 2 trials")).toBeVisible();

  await replayRecordedFailure(page, "authorization-boundary");

  await page.getByRole("button", { name: "Preview redacted AI evidence" }).click();
  await expect(page.getByRole("heading", { name: "Evidence preview" })).toBeVisible();
  await page.getByRole("button", { name: "Explicitly start AI Analysis" }).click();
  await expect(page.getByText(/Reused the existing analysis/)).toBeVisible();
  await expect(page.getByText("The authorization predicate is inverted.")).toBeVisible();

  await page.getByRole("button", { name: "Create bounded Handoff Bundle" }).click();
  await expect(page.getByLabel("MCP resource URI")).toHaveValue(
    "agentqa://workflows/workflow-t22/handoffs/handoff-t22",
  );
  await page.getByRole("button", { name: "Rerun preflight after repair" }).click();
  await expect(page.getByText("CHANGED", { exact: true }).first()).toBeVisible();

  await page.getByRole("button", { name: "Compare runs" }).click();
  await expect(page.getByText("QUALITY IMPROVED COST NEUTRAL")).toBeVisible();
  await page.getByRole("button", { name: "Evaluate regression gate" }).click();
  await expect(page.getByText("CLI-compatible exit code 0")).toBeVisible();
  const compareSection = page.locator(".assurance-section").filter({
    has: page.getByRole("heading", { name: "Transparent release decision" }),
  });
  await compareSection.locator("select").nth(2).selectOption("mutation");
  await page.getByLabel("GateSpec path").fill("agentqa/mutation-gate.yaml");
  await page.getByRole("button", { name: "Evaluate mutation gate" }).click();
  await expect(page.getByRole("heading", { name: "mutation gate" })).toBeVisible();
  await page.getByRole("button", { name: "Generate report" }).click();
  await expect(page.getByText("agentqa.workflow-report/v1alpha1", { exact: true })).toBeVisible();

  expect(calls).toContain("POST /api/workflows/workflow-t22/handoffs");
  expect(calls).toContain("POST /api/campaigns/campaign-replay-t22/start");
  expect(calls).toContain("POST /api/workflows/workflow-t22/comparisons");
  expect(calls.filter((value) => value === "POST /api/workflows/workflow-t22/gates")).toHaveLength(2);
  expect(calls).toContain("POST /api/workflows/workflow-t22/reports");
});
