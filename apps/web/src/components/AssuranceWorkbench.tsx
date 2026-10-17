import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api";
import { percent, shortId } from "../lib";
import type {
  AnalysisReport,
  CodeEvidence,
  ComparisonReport,
  FailureEvidencePreview,
  FailureInboxItem,
  HandoffBundle,
  StoredRunSummary,
  WorkflowGateResult,
  WorkflowReport,
  WorkflowRun,
} from "../types";
import { StatusBadge } from "./StatusBadge";

const changeText = (change: ComparisonReport["deltas"]["average_cost_usd"]) => {
  if (change.status === "unobservable") return "UNOBSERVABLE";
  if (change.status === "unbounded") return "UNBOUNDED";
  const value = change.relative_increase ?? 0;
  return `${value >= 0 ? "+" : ""}${percent(value)}`;
};

export function AssuranceWorkbench({
  workflow,
  codeEvidence,
  view = "investigate",
}: {
  workflow: WorkflowRun;
  codeEvidence: CodeEvidence[];
  view?: "investigate" | "release";
}) {
  const [failures, setFailures] = useState<FailureInboxItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [preview, setPreview] = useState<FailureEvidencePreview | null>(null);
  const [analyses, setAnalyses] = useState<AnalysisReport[]>([]);
  const [selectedHashes, setSelectedHashes] = useState<string[]>([]);
  const [reuseExisting, setReuseExisting] = useState(true);
  const [filters, setFilters] = useState({ risk: "", evaluator: "", tool: "", mutation: "", target: "", testRevision: "", analysisStatus: "" });
  const [runs, setRuns] = useState<StoredRunSummary[]>([]);
  const [baseline, setBaseline] = useState("");
  const [candidate, setCandidate] = useState("");
  const [comparison, setComparison] = useState<ComparisonReport | null>(null);
  const [gatePath, setGatePath] = useState("examples/specs/quality-gate.yaml");
  const [gateKind, setGateKind] = useState<"regression" | "mutation">("regression");
  const [gate, setGate] = useState<WorkflowGateResult | null>(null);
  const [handoff, setHandoff] = useState<HandoffBundle | null>(null);
  const [developerTask, setDeveloperTask] = useState("Investigate the cited failure, implement a bounded fix, and return a candidate target for regression and mutation testing.");
  const [repairChanges, setRepairChanges] = useState<Record<string, boolean | null> | null>(null);
  const [report, setReport] = useState<WorkflowReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    const activeFilters = Object.fromEntries(Object.entries(filters).filter(([, value]) => value));
    const [nextFailures, nextAnalyses, runResponse] = await Promise.all([
      api.listFailures(workflow.workflow_id, activeFilters),
      api.listAnalyses(workflow.workflow_id),
      api.listRuns(workflow.project_id),
    ]);
    setFailures(nextFailures);
    setAnalyses(nextAnalyses);
    const nextRuns = runResponse.runs ?? [];
    setRuns(nextRuns);
    setSelectedId((current) => current && nextFailures.some((item) => item.failure_id === current) ? current : nextFailures[0]?.failure_id ?? null);
    setBaseline((current) => current || nextRuns[1]?.run_id || nextRuns[0]?.run_id || "");
    setCandidate((current) => current || nextRuns[0]?.run_id || "");
  }, [filters, workflow.project_id, workflow.workflow_id]);

  useEffect(() => {
    setFailures([]); setAnalyses([]); setPreview(null); setSelectedId(null); setHandoff(null); setReport(null);
    void load().catch((reason) => setError((reason as Error).message));
  }, [load]);

  const selected = useMemo(() => failures.find((item) => item.failure_id === selectedId) ?? null, [failures, selectedId]);
  const selectedAnalysis = useMemo(() => analyses.find((item) => item.analysis_id === selected?.latest_analysis_id) ?? analyses.find((item) => item.run_id === selected?.run_id && item.trial_id === selected?.trial_id) ?? null, [analyses, selected]);

  const act = async (operation: () => Promise<void>) => {
    setBusy(true); setError(null); setNotice(null);
    try { await operation(); }
    catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const previewEvidence = () => act(async () => {
    if (!selectedId) return;
    setPreview(await api.previewFailureEvidence(workflow.workflow_id, selectedId, selectedHashes));
  });

  const analyze = () => act(async () => {
    if (!selectedId || !preview) return;
    const response = await api.analyzeFailure(workflow.workflow_id, selectedId, {
      idempotencyKey: `analysis-${Date.now()}`,
      codeEvidenceHashes: selectedHashes,
      reuseExisting,
    });
    if (response.reused) {
      setNotice("Reused the existing analysis because its evidence has not changed.");
    } else if (response.step) {
      setNotice("AI analysis started. Waiting for the persisted report…");
      for (let attempt = 0; attempt < 40; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        const current = await api.getWorkflow(workflow.workflow_id);
        const step = current.steps.find((item) => item.step_id === response.step?.step_id);
        if (step?.status === "COMPLETED" || step?.status === "FAILED_SAFE") break;
      }
    }
    await load();
  });

  const createHandoff = () => act(async () => {
    if (!selectedAnalysis) return;
    const response = await api.createHandoff(workflow.workflow_id, selectedAnalysis.analysis_id, developerTask);
    setHandoff(response.handoff);
    setNotice("Handoff Bundle created. The workflow is waiting for external repair.");
  });

  const repairPreflight = () => act(async () => {
    const response = await api.repairPreflight(workflow.workflow_id);
    setRepairChanges(response.changes);
    setNotice(response.report.executable ? "Repair preflight passed; select the new candidate in Campaign Builder and rerun." : "Repair preflight is blocked. Review its checks before rerunning.");
  });

  const compare = () => act(async () => {
    const response = await api.workflowCompare(workflow.workflow_id, baseline, candidate);
    setComparison(response.report);
  });

  const evaluateGate = () => act(async () => {
    const response = await api.workflowGate(workflow.workflow_id, {
      kind: gateKind,
      gatePath,
      candidateRunId: candidate,
      baselineRunId: gateKind === "regression" ? baseline : undefined,
    });
    setGate(response.gate);
  });

  const exportReport = () => act(async () => {
    const response = await api.exportWorkflowReport(workflow.workflow_id);
    setReport(response.report);
    setNotice(`WorkflowReport verified at ${response.artifact.uri}.`);
  });

  const downloadReport = () => {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `workflow-${workflow.workflow_id}-report.json`; anchor.click();
    URL.revokeObjectURL(url);
  };

  return <div className="assurance-workbench">
    <div className="panel__header"><div><p className="eyebrow">{view === "investigate" ? "Failure investigation" : "Release decision"}</p><h2>{view === "investigate" ? "Understand and repair failures" : "Compare, gate and report"}</h2></div><button className="button button--small" disabled={busy} onClick={() => void load()}>Refresh</button></div>
    {error && <div className="notice notice--bad">{error}</div>}
    {notice && <div className="notice">{notice}</div>}

    {view === "investigate" && <><section className="assurance-section">
      <div className="panel__header"><div><p className="eyebrow">1 · Failure Inbox</p><h3>Campaign failures</h3></div><strong>{failures.length}</strong></div>
      <div className="failure-filters">
        {(["risk", "evaluator", "tool", "mutation", "target", "testRevision"] as const).map((name) => <label key={name}>{name}<input value={filters[name]} onChange={(event) => setFilters((current) => ({ ...current, [name]: event.target.value }))} /></label>)}
        <label>analysis<select value={filters.analysisStatus} onChange={(event) => setFilters((current) => ({ ...current, analysisStatus: event.target.value }))}><option value="">all</option><option>NOT_ANALYZED</option><option>ANALYZING</option><option>ANALYZED</option><option>FAILED_SAFE</option></select></label>
      </div>
      <div className="failure-layout">
        <div className="failure-list">{failures.map((item) => <button key={item.failure_id} className={item.failure_id === selectedId ? "is-selected" : ""} onClick={() => { setSelectedId(item.failure_id); setPreview(null); }}><StatusBadge status={item.analysis_status} /><strong>{item.test_case_id}</strong><small>{item.target_role} · {item.risk} · {shortId(item.run_id)}</small><span>{item.failed_evaluators.join(", ") || "failed without evaluator detail"}</span></button>)}{failures.length === 0 && <p className="muted padded">No Campaign failures match these filters.</p>}</div>
        {selected && <div className="failure-detail"><div className="workflow-meta"><span>Trace<strong>{selected.trace_status}</strong></span><span>Tools<strong>{selected.tools.length}</strong></span><span>Mutations<strong>{selected.mutation_ids.length}</strong></span><span>Analyses<strong>{selected.analysis_count}</strong></span></div><p>{selected.failure_bundle_ref ? "A verified failure bundle is available." : "The failure bundle is unavailable."}</p>{selected.replay_cassette_ref ? <label>Replay recording<input aria-label="Replay cassette path" className="mono" readOnly value={selected.replay_cassette_ref} /></label> : <p>No replay recording is available.</p>}<div className="selected-evidence"><h4>Source context for AI</h4>{codeEvidence.length === 0 ? <p className="muted">Build repository context in Test design to give AI more implementation detail.</p> : codeEvidence.map((item) => <label className="check-row" key={item.contentHash}><input type="checkbox" checked={selectedHashes.includes(item.contentHash)} onChange={(event) => setSelectedHashes((current) => event.target.checked ? [...current, item.contentHash] : current.filter((value) => value !== item.contentHash))} /><span>{item.path}<small>lines {item.startLine}–{item.endLine}</small></span></label>)}</div><button className="button" disabled={busy} onClick={() => void previewEvidence()}>Review AI input</button></div>}
      </div>
    </section>

    {preview && <section className="assurance-section evidence-review"><div className="panel__header"><div><p className="eyebrow">AI input review</p><h3>Evidence selected for analysis</h3></div></div><div className="metric-grid metric-grid--compact"><article className="metric"><span>Evidence items</span><strong>{preview.bundle.items.length}</strong></article><article className="metric"><span>Missing</span><strong>{preview.bundle.missing_evidence.length}</strong></article><article className="metric"><span>Blocking gaps</span><strong>{preview.bundle.blocking_gaps.length}</strong></article><article className="metric"><span>Existing analyses</span><strong>{preview.reusable_analyses.length}</strong></article></div><div className="evidence-items">{preview.bundle.items.map((item) => <details key={item.evidence_id}><summary>{item.kind} · {item.summary}</summary><pre>{JSON.stringify(item.data, null, 2)}</pre></details>)}</div>{preview.bundle.missing_evidence.length > 0 && <div className="notice notice--warn">Missing evidence: {preview.bundle.missing_evidence.join(" · ")}</div>}<div className="button-row"><label className="check-row"><input type="checkbox" checked={reuseExisting} onChange={(event) => setReuseExisting(event.target.checked)} /><span>Reuse an existing analysis when inputs are unchanged</span></label><button className="button button--ai" disabled={busy || workflow.status !== "READY"} onClick={() => void analyze()}><span>✦</span>Analyze failure with AI</button></div></section>}

    {selectedAnalysis && <section className="assurance-section analysis-report"><div className="panel__header"><div><p className="eyebrow">3 · Non-authoritative diagnosis</p><h3>{selectedAnalysis.test_case_id}</h3></div><span className="status status--good">{selectedAnalysis.llm_call.status}</span></div><h4>Observed facts</h4>{selectedAnalysis.observed_facts.map((fact) => <article key={fact.statement}><p>{fact.statement}</p><code>{fact.evidence_ids.join(", ")}</code></article>)}<h4>Earliest divergence</h4><p>{selectedAnalysis.earliest_divergence?.description ?? "Not observable"}</p><h4>Root-cause candidates</h4>{selectedAnalysis.root_causes.map((cause) => <article className={cause.confidence < 0.6 ? "low-confidence" : ""} key={cause.id}><div><strong>{cause.hypothesis}</strong><StatusBadge status={cause.confidence < 0.6 ? "LOW CONFIDENCE" : "HYPOTHESIS"}>{percent(cause.confidence)}</StatusBadge></div><p>{cause.reasoning}</p><code>Evidence: {cause.evidence_ids.join(", ")}</code>{cause.confidence < 0.6 && <small>Low-confidence suggestion — not a confirmed fact.</small>}</article>)}<div className="analysis-columns"><div><h4>Missing evidence</h4><ul>{selectedAnalysis.missing_evidence.map((item) => <li key={item}>{item}</li>)}</ul></div><div><h4>Suggested tests</h4><ul>{selectedAnalysis.regression_tests.map((item) => <li key={item.title}><strong>{item.title}</strong> — {item.expected_behavior}</li>)}</ul></div></div><label>External repair task<textarea value={developerTask} onChange={(event) => setDeveloperTask(event.target.value)} /></label><button className="button" disabled={busy || workflow.status !== "READY"} onClick={() => void createHandoff()}>Create bounded Handoff Bundle</button></section>}

    {(handoff || workflow.status === "WAITING_EXTERNAL_REPAIR") && <section className="assurance-section"><div className="panel__header"><div><p className="eyebrow">Repair checkpoint</p><h3>Waiting for developer or Coding Agent</h3></div><StatusBadge status="WAITING_EXTERNAL_REPAIR" /></div>{handoff && <div className="handoff-copy"><label>MCP resource URI<input readOnly value={handoff.handoff_resource_uri} /></label><button className="button button--small" onClick={() => void navigator.clipboard?.writeText(handoff.handoff_resource_uri)}>Copy URI</button><label>Repair task<textarea readOnly value={handoff.developer_task} /></label><button className="button button--small" onClick={() => void navigator.clipboard?.writeText(handoff.developer_task)}>Copy task</button></div>}<button className="button" disabled={busy} onClick={() => void repairPreflight()}>Check repaired version</button>{repairChanges && <div className="change-grid">{Object.entries(repairChanges).map(([key, changed]) => <span key={key}>{key}<strong>{changed === null ? "UNOBSERVABLE" : changed ? "CHANGED" : "UNCHANGED"}</strong></span>)}</div>}</section>}</>}

    {view === "release" && <><section className="assurance-section"><div className="panel__header"><div><p className="eyebrow">Quality decision</p><h3>Compare versions and apply a release gate</h3></div></div><div className="form-grid"><label>Baseline<select value={baseline} onChange={(event) => setBaseline(event.target.value)}><option value="">Select run</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{shortId(run.run_id)} · {percent(run.summary.pass_rate)}</option>)}</select></label><label>Candidate<select value={candidate} onChange={(event) => setCandidate(event.target.value)}><option value="">Select run</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{shortId(run.run_id)} · {percent(run.summary.pass_rate)}</option>)}</select></label><label>Decision type<select value={gateKind} onChange={(event) => setGateKind(event.target.value as typeof gateKind)}><option value="regression">Regression</option><option value="mutation">Mutation strength</option></select></label><label>Release policy<input value={gatePath} onChange={(event) => setGatePath(event.target.value)} /></label></div><div className="button-row"><button className="button" disabled={busy || !baseline || !candidate || workflow.status !== "READY"} onClick={() => void compare()}>Compare runs</button><button className="button button--accent" disabled={busy || !candidate || !gatePath || workflow.status !== "READY"} onClick={() => void evaluateGate()}>Make release decision</button></div>{comparison && <><div className="metric-grid metric-grid--compact"><article className="metric"><span>Fixed</span><strong>{comparison.counts.fixed}</strong></article><article className="metric"><span>Regressed</span><strong>{comparison.counts.new_regressions}</strong></article><article className="metric"><span>Unchanged pass</span><strong>{comparison.counts.stable}</strong></article><article className="metric"><span>Incomplete</span><strong>{comparison.counts.new + comparison.counts.removed}</strong></article></div><div className="quality-cost"><div><StatusBadge status={comparison.quality_cost.status} /><strong>Quality / cost view</strong></div><div className="change-grid"><span>Pass rate<strong>{comparison.deltas.pass_rate >= 0 ? "+" : ""}{percent(comparison.deltas.pass_rate)}</strong></span><span>Average tokens<strong>{changeText(comparison.deltas.average_tokens)}</strong></span><span>Average cost<strong>{changeText(comparison.deltas.average_cost_usd)}</strong></span><span>P95 latency<strong>{changeText(comparison.deltas.p95_latency_ms)}</strong></span></div><ul>{comparison.quality_cost.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></div><div className="case-grid">{comparison.cases.map((item) => <article className="case-card" key={item.test_case_id}><StatusBadge status={item.classification} /><strong>{item.test_case_id}</strong></article>)}</div></>}{gate && <div className={gate.result.passed ? "gate-result gate-result--pass" : "gate-result gate-result--fail"}><div className="gate-verdict"><span>{gate.result.passed ? "PASS" : "FAIL"}</span><div><h3>{gate.kind} gate</h3><p>{gate.result.passed ? "Candidate meets the release policy." : "Candidate does not meet the release policy."}</p></div></div><div className="rule-list">{gate.result.rules.map((rule) => <article key={rule.name}><StatusBadge status={rule.passed ? "PASS" : "FAIL"} /><div><strong>{rule.name}</strong><p>{rule.message}</p></div></article>)}</div></div>}</section>

    <section className="assurance-section"><div className="panel__header"><div><p className="eyebrow">Shareable result</p><h3>Evaluation report</h3></div></div><div className="button-row"><button className="button button--accent" disabled={busy || workflow.status !== "READY"} onClick={() => void exportReport()}>Generate report</button><button className="button" disabled={!report} onClick={downloadReport}>Download JSON</button></div>{report && <><div className="workflow-meta"><span>Artifacts<strong>{report.authoritative_artifacts.length}</strong></span><span>Analyses<strong>{String(report.analyses.total ?? 0)}</strong></span><span>Approvals<strong>{String(report.approvals.count ?? 0)}</strong></span><span>Missing evidence<strong>{report.missing_evidence.length}</strong></span></div><details><summary>Inspect report data</summary><pre>{JSON.stringify(report, null, 2)}</pre></details></>}</section></>}
  </div>;
}
