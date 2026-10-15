import type {
  AIEvidenceResponse,
  ArtifactImportResult,
  Campaign,
  CampaignInputs,
  CatalogArtifact,
  CodeEvidence,
  ComparisonReport,
  EvidenceBoundaryPreview,
  GenerationSession,
  GenerationSessionDetail,
  GateResult,
  HealthResponse,
  PreflightReport,
  RunJob,
  RunsResponse,
  StoredFinding,
  StoredRunDetail,
  TestResult,
  TraceRecord,
  WorkbenchProject,
  WorkspaceEntry,
  WorkflowEvent,
  WorkflowRun,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  const payload = (await response.json().catch(() => ({}))) as {
    detail?: { code?: string; message?: string } | string;
  };
  if (!response.ok) {
    const detail = payload.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message ?? `Request failed with HTTP ${response.status}`;
    throw new ApiError(message, response.status, typeof detail === "object" ? detail.code : undefined);
  }
  return payload as T;
}

export const api = {
  aiEvidence() {
    return request<AIEvidenceResponse>("/api/ai/evidence");
  },
  health() {
    return request<HealthResponse>("/api/health");
  },
  async listProjects() {
    const result = await request<{ projects: WorkbenchProject[] }>("/api/projects");
    return result.projects;
  },
  createProject(input: {
    projectId: string;
    name: string;
    description?: string;
    workspacePath?: string;
    defaultSpecRefs?: Record<string, string>;
    tags?: string[];
  }) {
    return request<WorkbenchProject>("/api/projects", {
      method: "POST",
      body: JSON.stringify(input),
    });
  },
  getProject(projectId: string) {
    return request<WorkbenchProject>(`/api/projects/${encodeURIComponent(projectId)}`);
  },
  preflightProject(projectId: string) {
    return request<PreflightReport>(`/api/projects/${encodeURIComponent(projectId)}/preflight`, {
      method: "POST",
    });
  },
  importArtifacts(projectId: string, sources: string[]) {
    return request<ArtifactImportResult>(
      `/api/projects/${encodeURIComponent(projectId)}/artifacts/import`,
      { method: "POST", body: JSON.stringify({ sources }) },
    );
  },
  async listArtifacts(projectId: string) {
    const result = await request<{ artifacts: CatalogArtifact[]; total?: number }>(
      `/api/projects/${encodeURIComponent(projectId)}/artifacts`,
    );
    return { artifacts: result.artifacts, total: result.total ?? result.artifacts.length };
  },
  async browseWorkspace(projectId: string, path = ".") {
    const result = await request<{ entries: WorkspaceEntry[] }>(
      `/api/projects/${encodeURIComponent(projectId)}/workspace?path=${encodeURIComponent(path)}`,
    );
    return result.entries;
  },
  previewCodeEvidence(
    projectId: string,
    input: { path: string; startLine: number; endLine: number; reason: string },
  ) {
    return request<CodeEvidence>(
      `/api/projects/${encodeURIComponent(projectId)}/code-evidence/preview`,
      { method: "POST", body: JSON.stringify(input) },
    );
  },
  addCodeEvidence(
    projectId: string,
    input: { path: string; startLine: number; endLine: number; reason: string },
  ) {
    return request<CodeEvidence>(`/api/projects/${encodeURIComponent(projectId)}/code-evidence`, {
      method: "POST",
      body: JSON.stringify(input),
    });
  },
  async listCodeEvidence(projectId: string) {
    const result = await request<{ codeEvidence: CodeEvidence[] }>(
      `/api/projects/${encodeURIComponent(projectId)}/code-evidence`,
    );
    return result.codeEvidence;
  },
  evidencePreview(projectId: string) {
    return request<EvidenceBoundaryPreview>(
      `/api/projects/${encodeURIComponent(projectId)}/evidence-preview`,
    );
  },
  async listWorkflows(projectId?: string) {
    const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
    const result = await request<{ workflows: WorkflowRun[] }>(`/api/workflows${query}`);
    return result.workflows;
  },
  createWorkflow(input: {
    projectId: string;
    name: string;
    description?: string;
    idempotencyKey: string;
    draft?: boolean;
  }) {
    return request<WorkflowRun>("/api/workflows", {
      method: "POST",
      body: JSON.stringify(input),
    });
  },
  getWorkflow(workflowId: string) {
    return request<WorkflowRun>(`/api/workflows/${encodeURIComponent(workflowId)}`);
  },
  async listWorkflowEvents(workflowId: string, after = 0) {
    const result = await request<{ events: WorkflowEvent[]; revision: number }>(
      `/api/workflows/${encodeURIComponent(workflowId)}/events?after=${after}`,
    );
    return result;
  },
  resumeWorkflow(workflowId: string) {
    return request<WorkflowRun>(`/api/workflows/${encodeURIComponent(workflowId)}/resume`, {
      method: "POST",
    });
  },
  cancelWorkflow(workflowId: string) {
    return request<WorkflowRun>(`/api/workflows/${encodeURIComponent(workflowId)}/cancel`, {
      method: "POST",
    });
  },
  async listGenerationSessions(workflowId: string) {
    const result = await request<{ sessions: GenerationSession[] }>(
      `/api/workflows/${encodeURIComponent(workflowId)}/generation-sessions`,
    );
    return result.sessions ?? [];
  },
  createGenerationSession(workflowId: string, input: Record<string, unknown>) {
    return request<{ session: GenerationSession; step: unknown }>(
      `/api/workflows/${encodeURIComponent(workflowId)}/generation-sessions`,
      { method: "POST", body: JSON.stringify(input) },
    );
  },
  getGenerationSession(sessionId: string) {
    return request<GenerationSessionDetail>(
      `/api/generation-sessions/${encodeURIComponent(sessionId)}`,
    );
  },
  generateCandidates(sessionId: string, planChecksum: string, idempotencyKey: string) {
    return request<{ session: GenerationSession; step: unknown }>(
      `/api/generation-sessions/${encodeURIComponent(sessionId)}/candidates`,
      { method: "POST", body: JSON.stringify({ planChecksum, idempotencyKey }) },
    );
  },
  reviewCandidate(
    sessionId: string,
    caseId: string,
    input: { subjectHash: string; decision: string; reviewer: string; comment: string },
  ) {
    return request(
      `/api/generation-sessions/${encodeURIComponent(sessionId)}/candidates/${encodeURIComponent(caseId)}/review`,
      { method: "POST", body: JSON.stringify(input) },
    );
  },
  batchApproveCandidates(sessionId: string, caseIds: string[], reviewer: string, comment: string) {
    return request(
      `/api/generation-sessions/${encodeURIComponent(sessionId)}/candidates/batch-approve`,
      { method: "POST", body: JSON.stringify({ caseIds, reviewer, comment }) },
    );
  },
  reviseCandidate(
    sessionId: string,
    caseId: string,
    input: { spec: Record<string, unknown>; editor: string; diffSummary: string },
  ) {
    return request(
      `/api/generation-sessions/${encodeURIComponent(sessionId)}/candidates/${encodeURIComponent(caseId)}/revisions`,
      { method: "POST", body: JSON.stringify(input) },
    );
  },
  campaignInputs(workflowId: string) {
    return request<CampaignInputs>(
      `/api/workflows/${encodeURIComponent(workflowId)}/campaign-inputs`,
    );
  },
  async listCampaigns(workflowId: string) {
    const result = await request<{ campaigns: Campaign[] }>(
      `/api/workflows/${encodeURIComponent(workflowId)}/campaigns`,
    );
    return result.campaigns ?? [];
  },
  createCampaign(workflowId: string, input: Record<string, unknown>) {
    return request<Campaign>(`/api/workflows/${encodeURIComponent(workflowId)}/campaigns`, {
      method: "POST",
      body: JSON.stringify(input),
    });
  },
  createCampaignChallenge(campaignId: string) {
    return request<{
      challenge: { challenge_id: string; config_hash: string; expires_at: string };
      phrase: string;
      risk: Campaign["risk"];
    }>(`/api/campaigns/${encodeURIComponent(campaignId)}/challenge`, { method: "POST" });
  },
  confirmCampaign(
    campaignId: string,
    input: { challengeId: string; phrase: string; configHash: string; reviewer: string; comment: string },
  ) {
    return request<Campaign>(`/api/campaigns/${encodeURIComponent(campaignId)}/confirm`, {
      method: "POST",
      body: JSON.stringify(input),
    });
  },
  startCampaign(campaignId: string, configHash: string, idempotencyKey: string) {
    return request<{ campaign: Campaign; step: unknown }>(
      `/api/campaigns/${encodeURIComponent(campaignId)}/start`,
      { method: "POST", body: JSON.stringify({ configHash, idempotencyKey }) },
    );
  },
  listRuns(projectId = "default") {
    return request<RunsResponse>(`/api/runs?project_id=${encodeURIComponent(projectId)}`);
  },
  createRun(input: {
    projectId: string;
    testsPath: string;
    targetPath?: string;
    mutationsPath?: string;
    manifestPath?: string;
    judgeEnabled?: boolean;
  }) {
    return request<RunJob>("/api/runs", { method: "POST", body: JSON.stringify(input) });
  },
  async getRun(runId: string) {
    const result = await request<{ state: string; storedRun?: StoredRunDetail }>(
      `/api/runs/${encodeURIComponent(runId)}`,
    );
    return result.storedRun ?? null;
  },
  async getResults(runId: string) {
    const result = await request<{ results: TestResult[] }>(
      `/api/runs/${encodeURIComponent(runId)}/results`,
    );
    return result.results;
  },
  async getFindings(runId: string) {
    const result = await request<{ findings: StoredFinding[] }>(
      `/api/runs/${encodeURIComponent(runId)}/findings`,
    );
    return result.findings;
  },
  getTrace(traceId: string) {
    return request<TraceRecord>(`/api/traces/${encodeURIComponent(traceId)}`);
  },
  updateReview(findingId: string, status: string, note = "") {
    return request(`/api/findings/${encodeURIComponent(findingId)}/review`, {
      method: "PUT",
      body: JSON.stringify({ status, note }),
    });
  },
  compare(baselineRunId: string, candidateRunId: string) {
    return request<ComparisonReport>("/api/comparisons", {
      method: "POST",
      body: JSON.stringify({ baselineRunId, candidateRunId }),
    });
  },
  gate(gatePath: string, candidateRunId: string, baselineRunId?: string) {
    return request<GateResult>("/api/gates/evaluate", {
      method: "POST",
      body: JSON.stringify({ gatePath, candidateRunId, baselineRunId }),
    });
  },
};
