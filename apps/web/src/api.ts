import type {
  ComparisonReport,
  GateResult,
  HealthResponse,
  RunJob,
  RunsResponse,
  StoredFinding,
  TestResult,
  TraceRecord,
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
  health() {
    return request<HealthResponse>("/api/health");
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
