export type RunStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";

export interface MetricSummary {
  count: number;
  mean: number | null;
  p50: number | null;
  p95: number | null;
}

export interface CaseSummary {
  test_case_id: string;
  passed: boolean | null;
  pass_rate: number;
  completed_trials: number;
  planned_trials: number;
  inconsistent: boolean;
  latency_ms: MetricSummary;
  total_tokens: MetricSummary;
}

export interface RunSummary {
  total: number;
  passed: number;
  failed: number;
  incomplete: number;
  pass_rate: number;
  inconsistent_cases: number;
  cases: CaseSummary[];
}

export interface StoredRunSummary {
  project_id: string;
  run_id: string;
  status: RunStatus;
  exit_code: number;
  started_at: string;
  ended_at: string;
  summary: RunSummary;
  mutation_score: number | null;
  artifact_path: string | null;
  finding_count: number;
}

export interface RunJob {
  runId: string;
  projectId: string;
  status: RunStatus;
  submittedAt: string;
  finishedAt: string | null;
  error: { code: string; message: string } | null;
}

export interface RunsResponse {
  runs: StoredRunSummary[];
  jobs: RunJob[];
}

export interface HealthResponse {
  status: string;
  storage: string;
  workflowStorage: string;
  workspaceRoot: string;
  judge: {
    configured: boolean;
    provider: string | null;
    model: string | null;
  };
}

export type WorkflowStatus =
  | "DRAFT"
  | "READY"
  | "RUNNING"
  | "WAITING_REVIEW"
  | "WAITING_EXTERNAL_REPAIR"
  | "INTERRUPTED"
  | "FAILED_SAFE"
  | "COMPLETED"
  | "CANCELLED";

export interface WorkbenchProject {
  project_id: string;
  name: string;
  description: string;
  workspace_path: string;
  default_spec_refs: Record<string, string>;
  tags: string[];
  recent_workflow_id: string | null;
  last_manifest_snapshot: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface PreflightCheck {
  check_id: string;
  category: string;
  status: "PASS" | "REVIEW" | "BLOCK";
  message: string;
  path: string | null;
}

export interface PreflightReport {
  project_id: string;
  checked_at: string;
  executable: boolean;
  requires_approval: boolean;
  checks: PreflightCheck[];
  manifest_snapshot: Record<string, unknown> | null;
}

export interface CatalogArtifact {
  artifact_id: string;
  project_id: string;
  artifact_type: string;
  source: string;
  path: string;
  schema_version: string;
  checksum: string;
  size_bytes: number;
  created_at: string;
  workflow_id: string | null;
  run_id: string | null;
  availability: "AVAILABLE" | "MISSING" | "INVALID";
  metadata: Record<string, unknown>;
}

export interface ArtifactImportResult {
  project_id: string;
  scanned: number;
  imported: number;
  unchanged: number;
  rejected: number;
  runs_registered: number;
  issues: Array<{ path: string; code: string; message: string }>;
}

export interface WorkspaceEntry {
  path: string;
  name: string;
  kind: "file" | "directory";
  size_bytes: number | null;
}

export interface CodeEvidence {
  path: string;
  startLine: number;
  endLine: number;
  contentHash: string;
  reason: string;
  content: string;
}

export interface EvidenceBoundaryPreview {
  project_id: string;
  manifest_sections: string[];
  code_evidence: CodeEvidence[];
  evidence_types: string[];
  excluded: string[];
  provider: string | null;
  model: string | null;
  max_input_tokens: number;
  max_output_tokens: number;
  timeout_seconds: number;
  max_cost_usd: number | null;
  evidence_hash: string;
}

export interface WorkflowStep {
  step_id: string;
  workflow_id: string;
  kind: string;
  status: string;
  attempt: number;
  idempotency_key: string;
  input_hash: string;
  input_data: Record<string, unknown>;
  artifacts: Array<{
    artifact_id: string;
    artifact_type: string;
    uri: string;
    schema_version: string;
    checksum: string;
    source: string;
    verification_status: "VERIFIED" | "UNVERIFIED" | "INVALID";
  }>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: { code: string; message: string } | null;
}

export interface WorkflowRun {
  workflow_id: string;
  project_id: string;
  name: string;
  description: string;
  status: WorkflowStatus;
  revision: number;
  idempotency_key: string;
  input_hash: string;
  input_snapshot: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: { code: string; message: string } | null;
  steps: WorkflowStep[];
}

export interface WorkflowEvent {
  event_id: string;
  workflow_id: string;
  revision: number;
  event_type: string;
  occurred_at: string;
  message: string;
  step_id: string | null;
  data: Record<string, unknown>;
}

export interface Violation {
  type: string;
  message: string;
  span_id: string | null;
  details: Record<string, unknown>;
}

export interface MutationApplication {
  mutation_id: string;
  operation_type: string;
  triggered: boolean;
  call_index: number;
  behavior: Record<string, unknown>;
}

export interface TestResult {
  test_case_id: string;
  trial_id: string;
  trace_id: string;
  trial_index: number;
  passed: boolean;
  started_at: string;
  ended_at: string;
  trace_status: "complete" | "partial" | "missing";
  target_result: {
    status: "OK" | "ERROR" | "TIMEOUT";
    output: unknown;
    metadata: Record<string, unknown>;
    error: { code: string; message: string } | null;
  };
  evaluations: Array<{
    evaluator_type: string;
    status: string;
    score: number | null;
    message: string | null;
    metadata: Record<string, unknown>;
  }>;
  violations: Violation[];
  mutation_applications: MutationApplication[];
}

export interface StoredFinding {
  finding_id: string;
  run_id: string;
  test_case_id: string;
  trial_id: string;
  type: string;
  message: string;
  span_id: string | null;
  details: Record<string, unknown>;
}

export interface TraceEvent {
  trace_id: string;
  span_id: string;
  parent_span_id: string | null;
  kind: string;
  name: string;
  started_at: string;
  ended_at: string;
  status: string;
  attributes: Record<string, unknown>;
  input: unknown;
  output: unknown;
}

export interface TraceRecord {
  trace_id: string;
  status: "complete" | "partial" | "missing";
  root_span_id: string | null;
  events: TraceEvent[];
  diagnostics: string[];
  dropped_spans: number;
}

export interface CaseComparison {
  test_case_id: string;
  classification: string;
  baseline_passed: boolean | null;
  candidate_passed: boolean | null;
  risk: string | null;
}

export interface ComparisonReport {
  baseline_run_id: string;
  candidate_run_id: string;
  candidate_effective_pass_rate: number;
  counts: {
    new_regressions: number;
    fixed: number;
    persistent_failures: number;
    stable: number;
    new: number;
    removed: number;
  };
  baseline: { pass_rate: number; mutation_score: number | null };
  candidate: { pass_rate: number; mutation_score: number | null };
  cases: CaseComparison[];
}

export interface GateResult {
  passed: boolean;
  exit_code: number;
  rules: Array<{
    name: string;
    passed: boolean;
    actual: unknown;
    expected: unknown;
    message: string | null;
  }>;
}

export interface AIEvidenceResponse {
  generations: Array<{
    generation_id: string;
    path: string;
    generated_at: string;
    manifest_hash: string;
    plan_hash: string;
    context_hashes: string[];
    coverage_gaps: number;
    cases: Array<{
      case_id: string;
      risk: string;
      review_status: "READY" | "REVIEW_REQUIRED" | "APPROVED";
      source_refs: string[];
    }>;
    llm_calls: Array<{
      use_case: string;
      provider: string;
      model: string;
      status: string;
      attempts: number;
      token_usage: Record<string, number>;
      latency_ms: number | null;
      evidence_hash: string;
    }>;
  }>;
  analyses: Array<{
    analysis_id: string;
    path: string;
    generated_at: string;
    run_id: string;
    test_case_id: string;
    trial_id: string;
    evidence_hash: string;
    missing_evidence: string[];
    root_causes: Array<{
      id: string;
      hypothesis: string;
      confidence: number;
      reasoning: string;
      evidence_ids: string[];
    }>;
    llm_call: {
      provider: string;
      model: string;
      status: string;
      token_usage: Record<string, number>;
    };
  }>;
}

export type GenerationSessionStatus =
  | "PLANNING"
  | "PLAN_READY"
  | "GENERATING"
  | "CANDIDATES_READY"
  | "FAILED_SAFE";

export interface GenerationSession {
  session_id: string;
  workflow_id: string;
  project_id: string;
  status: GenerationSessionStatus;
  idempotency_key: string;
  input_hash: string;
  goal: {
    categories: string[];
    risk_levels: string[];
    target_capabilities: string[];
    target_policies: string[];
    allowed_mutations: string[];
    max_cases: number;
    instructions: string | null;
    code_evidence_hashes: string[];
  };
  evidence_mode: "MANIFEST_ONLY" | "MANIFEST_AND_CODE";
  code_evidence_hashes: string[];
  max_output_tokens: number;
  max_total_tokens: number | null;
  max_cost_usd: number | null;
  consumed_tokens: number;
  consumed_cost_usd: number | null;
  plan_path: string | null;
  plan_checksum: string | null;
  report_path: string | null;
  report_checksum: string | null;
  llm_calls: Array<{
    provider: string;
    model: string;
    status: string;
    token_usage: Record<string, number>;
    latency_ms: number | null;
    evidence_hash: string;
  }>;
  error: { code: string; message: string } | null;
  created_at: string;
  updated_at: string;
}

export interface GenerationPlan {
  generation_id: string;
  manifest_hash: string;
  goal_hash: string;
  items: Array<{
    id: string;
    title: string;
    category: string;
    risk: string;
    capabilities: string[];
    policies: string[];
    tools: string[];
    mutation: string | null;
    rationale: string;
    source_refs: string[];
    provenance: string[];
    requires_human: boolean;
  }>;
  questions: string[];
  warnings: string[];
}

export interface ApprovalDecision {
  approval_id: string;
  subject_hash: string;
  decision: "APPROVED" | "REJECTED" | "REQUEST_EDIT";
  reviewer: string;
  comment: string;
  created_at: string;
}

export interface GeneratedCandidate {
  case_id: string;
  risk: "low" | "medium" | "high" | "critical";
  review_status: string;
  rationale: string;
  assertion_basis: string[];
  coverage_item_ids: string[];
  source_refs: string[];
  original_spec_hash: string;
  current_spec_hash: string;
  spec: Record<string, unknown>;
  latest_revision: null | {
    revision_id: string;
    revision: number;
    parent_hash: string;
    spec_hash: string;
    editor: string;
    diff_summary: string;
    created_at: string;
  };
  decision: ApprovalDecision | null;
  diagnostics: Array<{ code: string; status: "PASS" | "WARN" | "BLOCK"; message: string }>;
  similarity_hints: Array<{ source: string; score: number }>;
  campaign_eligible: boolean;
}

export interface GenerationSessionDetail {
  session: GenerationSession;
  plan: GenerationPlan | null;
  candidates: GeneratedCandidate[];
}
