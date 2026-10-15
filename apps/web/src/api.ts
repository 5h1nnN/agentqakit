import type {
  AIEvidenceResponse,
  ComparisonReport,
  GateResult,
  HealthResponse,
  RunJob,
  RunsResponse,
  StoredFinding,
  TestResult,
  TraceRecord,
  WorkbenchProject,
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
  createProject(input: { projectId: string; name: string; description?: string }) {
    return request<WorkbenchProject>("/api/projects", {
      method: "POST",
      body: JSON.stringify(input),
    });
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
  getRun(runId: string) {
    return request<Record<string, unknown>>(`/api/runs/${encodeURIComponent(runId)}`);
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
