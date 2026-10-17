import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";

import { api } from "../api";
import type {
  ArtifactImportResult,
  CatalogArtifact,
  CodeEvidence,
  EvidenceBoundaryPreview,
  PreflightReport,
  RunJob,
  StoredRunSummary,
  WorkbenchProject,
  WorkflowEvent,
  WorkflowRun,
} from "../types";
import { AssuranceWorkbench } from "./AssuranceWorkbench";
import { CampaignWorkbench } from "./CampaignWorkbench";
import { GenerationWorkbench } from "./GenerationWorkbench";
import { RunDetail } from "./RunDetail";
import { RunList } from "./RunList";

type Phase = "overview" | "design" | "run" | "analyze" | "release";

const PHASES: Array<{ id: Phase; label: string; hint: string }> = [
  { id: "overview", label: "Overview", hint: "Setup and readiness" },
  { id: "design", label: "Test design", hint: "AI-generated coverage" },
  { id: "run", label: "Execute", hint: "Campaigns and results" },
  { id: "analyze", label: "Investigate", hint: "Failures and AI diagnosis" },
  { id: "release", label: "Release", hint: "Compare, gate, report" },
];

const REF_LABELS: Record<string, string> = {
  manifest: "Agent description",
  target: "Default agent",
  tests: "Test suite",
  mutations: "Mutation rules",
  gate: "Release policy",
  generation_goal: "Test-generation goal",
  baseline_target: "Comparison baseline",
  candidate_target: "Comparison candidate",
  defect_target: "Known-defect variant",
  fixed_target: "Fixed variant",
};

function statusClass(status: string) {
  if (["COMPLETED", "PASS", "AVAILABLE"].includes(status)) return "status--good";
  if (["FAILED_SAFE", "FAILED", "CANCELLED", "BLOCK", "INVALID", "MISSING"].includes(status)) return "status--bad";
  if (status === "RUNNING" || status === "REVIEW" || status.startsWith("WAITING")) return "status--warn";
  return "status--neutral";
}

function slug(value: string) {
  return value.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60);
}

export function Workbench() {
  const [projects, setProjects] = useState<WorkbenchProject[]>([]);
  const [workflows, setWorkflows] = useState<WorkflowRun[]>([]);
  const [projectId, setProjectId] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("overview");
  const [showCreate, setShowCreate] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [newProjectId, setNewProjectId] = useState("");
  const [workspacePath, setWorkspacePath] = useState(".");
  const [description, setDescription] = useState("");
  const [preflight, setPreflight] = useState<PreflightReport | null>(null);
  const [discoveryNotice, setDiscoveryNotice] = useState<string | null>(null);
  const [codeEvidence, setCodeEvidence] = useState<CodeEvidence[]>([]);
  const [boundary, setBoundary] = useState<EvidenceBoundaryPreview | null>(null);
  const [artifacts, setArtifacts] = useState<CatalogArtifact[]>([]);
  const [importResult, setImportResult] = useState<ArtifactImportResult | null>(null);
  const [runs, setRuns] = useState<StoredRunSummary[]>([]);
  const [jobs, setJobs] = useState<RunJob[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventCursor = useRef(0);

  const selectedProject = useMemo(
    () => projects.find((item) => item.project_id === projectId) ?? null,
    [projectId, projects],
  );
  const selected = useMemo(
    () => workflows.find((item) => item.workflow_id === selectedId) ?? null,
    [selectedId, workflows],
  );
  const selectedRun = useMemo(
    () => runs.find((item) => item.run_id === selectedRunId) ?? null,
    [runs, selectedRunId],
  );

  const refresh = useCallback(async () => {
    try {
      const nextProjects = await api.listProjects();
      const activeProjectId = projectId || nextProjects[0]?.project_id || "";
      const [nextWorkflows, runResponse] = await Promise.all([
        api.listWorkflows(activeProjectId || undefined),
        activeProjectId ? api.listRuns(activeProjectId) : Promise.resolve({ runs: [], jobs: [] }),
      ]);
      setProjects(nextProjects);
      setWorkflows(nextWorkflows);
      setRuns(runResponse.runs ?? []);
      setJobs(runResponse.jobs ?? []);
      if (!projectId && activeProjectId) setProjectId(activeProjectId);
      if (!selectedId && nextWorkflows.length > 0) setSelectedId(nextWorkflows[0].workflow_id);
      if (selectedId) {
        const timeline = await api.listWorkflowEvents(selectedId, eventCursor.current);
        eventCursor.current = timeline.revision;
        if (timeline.events.length) {
          setEvents((current) => {
            const known = new Set(current.map((item) => item.event_id));
            return [...current, ...timeline.events.filter((item) => !known.has(item.event_id))];
          });
        }
      }
      setError(null);
    } catch (reason) {
      setError((reason as Error).message);
    }
  }, [projectId, selectedId]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2500);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    eventCursor.current = 0;
    setEvents([]);
  }, [selectedId]);

  useEffect(() => {
    setPreflight(null);
    setDiscoveryNotice(null);
    setCodeEvidence([]);
    setBoundary(null);
    setArtifacts([]);
    setSelectedRunId(null);
  }, [projectId]);

  const createProject = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setError(null);
    try {
      const project = await api.createProject({
        projectId: newProjectId || slug(projectName),
        name: projectName,
        description,
        workspacePath,
        autoDiscover: true,
      });
      const workflow = await api.createWorkflow({
        projectId: project.project_id,
        name: "Main evaluation",
        description: "Primary quality workflow",
        idempotencyKey: `web-initial-${Date.now()}`,
      });
      setProjectId(project.project_id);
      setSelectedId(workflow.workflow_id);
      setProjectName(""); setNewProjectId(""); setDescription("");
      setShowCreate(false); setPhase("overview");
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const createWorkflow = async () => {
    if (!projectId) return;
    setBusy(true); setError(null);
    try {
      const workflow = await api.createWorkflow({
        projectId,
        name: `Evaluation ${workflows.length + 1}`,
        idempotencyKey: `web-workflow-${Date.now()}`,
      });
      setSelectedId(workflow.workflow_id);
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const scanProject = async () => {
    if (!projectId) return;
    setBusy(true); setError(null);
    try {
      const result = await api.discoverProject(projectId);
      setProjects((current) => current.map((item) => item.project_id === projectId ? result.project : item));
      const report = await api.preflightProject(projectId);
      setPreflight(report);
      setDiscoveryNotice(result.ready
        ? `Project is ready. ${Object.keys(result.project.default_spec_refs).length} resources were detected.`
        : `Setup needs attention: ${result.missing.join(", ")} could not be detected.`);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const buildContext = async () => {
    if (!projectId) return;
    setBusy(true); setError(null);
    try {
      const result = await api.discoverCodeEvidence(projectId);
      setCodeEvidence(await api.listCodeEvidence(projectId));
      setBoundary(await api.evidencePreview(projectId));
      setDiscoveryNotice(`${result.count} relevant source files are now available to AI features.`);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const importArtifacts = async () => {
    if (!projectId) return;
    setBusy(true); setError(null);
    try {
      setImportResult(await api.importArtifacts(projectId, ["artifacts", "mcp", "benchmark_results", "reference_results"]));
      const catalog = await api.listArtifacts(projectId);
      setArtifacts(catalog.artifacts);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const resume = async () => {
    if (!selected) return;
    setBusy(true); setError(null);
    try { await api.resumeWorkflow(selected.workflow_id); await refresh(); }
    catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const chooseProject = (value: string) => {
    setProjectId(value); setSelectedId(null); setPhase("overview");
  };

  return (
    <div className="workspace-shell">
      <aside className="project-rail">
        <div className="project-rail__heading"><span>Projects</span><button aria-label="New project" onClick={() => setShowCreate((value) => !value)}>+</button></div>
        <div className="project-list">
          {projects.map((project) => <button key={project.project_id} className={project.project_id === projectId ? "is-selected" : ""} onClick={() => chooseProject(project.project_id)}><span>{project.name.slice(0, 1).toUpperCase()}</span><div><strong>{project.name}</strong><small>{project.workspace_path}</small></div></button>)}
        </div>
        {projects.length === 0 && <p className="rail-empty">Create a project to begin.</p>}
        <button className="button button--small rail-create" onClick={() => setShowCreate(true)}>New project</button>
      </aside>

      <div className="workspace-main">
        {error && <div className="notice notice--bad global-notice">{error}</div>}
        {showCreate && <form className="panel quick-project" onSubmit={(event) => void createProject(event)}>
          <div><p className="eyebrow">New project</p><h1>Connect an agent repository</h1><p>Choose the repository folder. AgentQAKit will locate its manifest, target and tests automatically.</p></div>
          <div className="quick-project__fields"><label>Project name<input value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="Campus Ops Agent" required /></label><label>Repository directory<input className="mono" value={workspacePath} onChange={(event) => setWorkspacePath(event.target.value)} placeholder="examples/my-agent" required /></label></div>
          <details><summary>Optional details</summary><div className="form-grid form-grid--compact"><label>Project ID<input value={newProjectId} onChange={(event) => setNewProjectId(event.target.value)} placeholder={slug(projectName) || "generated-from-name"} /></label><label>Description<input value={description} onChange={(event) => setDescription(event.target.value)} /></label></div></details>
          <div className="button-row"><button type="button" className="button" onClick={() => setShowCreate(false)}>Cancel</button><button className="button button--accent" disabled={busy || !projectName || !workspacePath}>Create and scan</button></div>
        </form>}

        {!selectedProject && !showCreate && <section className="workspace-welcome"><div className="workspace-welcome__mark">AQ</div><p className="eyebrow">Agent quality workspace</p><h1>Test an agent from idea to release.</h1><p>Connect a repository once, then design tests, run campaigns, investigate failures and make a release decision in one place.</p><button className="button button--accent" onClick={() => setShowCreate(true)}>Create first project</button></section>}

        {selectedProject && <>
          <header className="workspace-header">
            <div><p className="eyebrow">{selectedProject.workspace_path}</p><h1>{selectedProject.name}</h1><p>{selectedProject.description || "Agent evaluation workspace"}</p></div>
            <div className="workspace-header__actions"><span className={`status ${statusClass(selected?.status ?? "DRAFT")}`}>{selected?.status ?? "NO WORKFLOW"}</span><button className="button button--small" onClick={() => void createWorkflow()} disabled={busy}>New evaluation</button></div>
          </header>

          <nav className="phase-nav" aria-label="Evaluation phases">
            {PHASES.map((item, index) => <button key={item.id} className={phase === item.id ? "is-active" : ""} disabled={!selected && item.id !== "overview"} onClick={() => setPhase(item.id)}><span>{index + 1}</span><div><strong>{item.label}</strong><small>{item.hint}</small></div></button>)}
          </nav>

          {discoveryNotice && <div className="notice workspace-notice">{discoveryNotice}</div>}

          {phase === "overview" && <div className="phase-content overview-grid">
            <section className="panel readiness-card">
              <div className="panel__header"><div><p className="eyebrow">Project setup</p><h2>Readiness</h2></div><button className="button button--accent" disabled={busy} onClick={() => void scanProject()}>{busy ? "Scanning…" : "Scan project"}</button></div>
              <div className="readiness-body">
                <div className={`readiness-score ${preflight?.executable ? "is-ready" : ""}`}><strong>{preflight ? (preflight.executable ? "Ready" : "Needs setup") : "Not checked"}</strong><span>{preflight ? `${preflight.checks.filter((item) => item.status === "PASS").length}/${preflight.checks.length} checks passed` : "Run a scan to validate configuration"}</span></div>
                <div className="detected-config">
                  {Object.entries(selectedProject.default_spec_refs).map(([kind, path]) => <article key={kind}><span className="config-dot" /><div><strong>{REF_LABELS[kind] ?? kind.replaceAll("_", " ")}</strong><small>{path}</small></div>{["manifest", "target", "tests"].includes(kind) && <em>required</em>}</article>)}
                  {Object.keys(selectedProject.default_spec_refs).length === 0 && <p className="muted">No AgentQA configuration detected yet.</p>}
                </div>
                {preflight && <details className="technical-details"><summary>Readiness details</summary><div className="check-list">{preflight.checks.map((check) => <article key={check.check_id}><span className={`status ${statusClass(check.status)}`}>{check.status}</span><div><strong>{check.message}</strong>{check.path && <small>{check.path}</small>}</div></article>)}</div></details>}
              </div>
            </section>

            <section className="panel evaluation-picker">
              <div className="panel__header"><div><p className="eyebrow">Evaluations</p><h2>Current work</h2></div></div>
              <div className="workflow-items">{workflows.map((workflow) => <button key={workflow.workflow_id} className={workflow.workflow_id === selectedId ? "workflow-item is-selected" : "workflow-item"} onClick={() => setSelectedId(workflow.workflow_id)}><span><strong>{workflow.name}</strong><small>{new Date(workflow.updated_at).toLocaleString()}</small></span><span className={`status ${statusClass(workflow.status)}`}>{workflow.status}</span></button>)}</div>
              {workflows.length === 0 && <div className="empty-state"><h2>No evaluation yet</h2><button className="button button--accent" onClick={() => void createWorkflow()}>Start evaluation</button></div>}
              {selected && <div className="evaluation-summary"><div><span>Status</span><strong>{selected.status}</strong></div><div><span>Completed actions</span><strong>{selected.steps.filter((item) => item.status === "COMPLETED").length}</strong></div><div><span>Last updated</span><strong>{new Date(selected.updated_at).toLocaleString()}</strong></div>{selected.error && <div className="notice notice--bad"><strong>{selected.error.code}</strong><span>{selected.error.message}</span></div>}{["DRAFT", "INTERRUPTED", "WAITING_EXTERNAL_REPAIR"].includes(selected.status) && <button className="button" disabled={busy} onClick={() => void resume()}>Resume evaluation</button>}</div>}
            </section>

            <section className="panel history-card">
              <div className="panel__header"><div><p className="eyebrow">Project data</p><h2>History and artifacts</h2></div><button className="button button--small" disabled={busy} onClick={() => void importArtifacts()}>Refresh history</button></div>
              <div className="history-summary"><span><strong>{runs.length}</strong> runs</span><span><strong>{artifacts.length}</strong> indexed artifacts</span><span><strong>{events.length}</strong> workflow events</span></div>
              {importResult && <p className="padded muted">Found {importResult.scanned} items; {importResult.imported} new items were indexed.</p>}
              {(artifacts.length > 0 || events.length > 0) && <details className="technical-details"><summary>Browse technical history</summary><div className="history-list">{events.slice(-10).reverse().map((item) => <article key={item.event_id}><strong>{item.event_type.replaceAll("_", " ")}</strong><span>{item.message}</span><small>{new Date(item.occurred_at).toLocaleString()}</small></article>)}{artifacts.slice(0, 10).map((item) => <article key={item.artifact_id}><strong>{item.artifact_type}</strong><span>{item.path}</span><small>{item.availability}</small></article>)}</div></details>}
            </section>
          </div>}

          {phase === "design" && <div className="phase-content">
            <section className="panel context-card">
              <div className="context-card__copy"><span className="ai-orb">✦</span><div><p className="eyebrow">Repository context</p><h2>Let AI understand the agent</h2><p>AgentQAKit scans the repository, selects relevant implementation files and redacts likely secrets. You can review the selected files before generating tests.</p></div></div>
              <button className="button button--ai" disabled={busy} onClick={() => void buildContext()}><span>✦</span>{codeEvidence.length ? "Refresh AI context" : "Build AI context"}</button>
              {codeEvidence.length > 0 && <div className="context-files">{codeEvidence.map((item) => <article key={item.contentHash}><strong>{item.path}</strong><small>lines {item.startLine}–{item.endLine}</small></article>)}</div>}
              {boundary && <details className="technical-details"><summary>What AI can access</summary><div className="boundary-summary"><p><strong>Included:</strong> agent manifest and {boundary.code_evidence.length} selected source files.</p><p><strong>Excluded:</strong> {boundary.excluded.join(", ")}.</p></div></details>}
            </section>
            {selected ? <GenerationWorkbench project={selectedProject} workflow={selected} codeEvidence={codeEvidence} /> : <div className="empty-state panel"><h2>Start an evaluation first</h2></div>}
          </div>}

          {phase === "run" && <div className="phase-content">
            {selected && <CampaignWorkbench workflow={selected} />}
            {selectedRun ? <><button className="back-button" onClick={() => setSelectedRunId(null)}>← All results</button><RunDetail run={selectedRun} /></> : <RunList runs={runs} jobs={jobs} selectedRunId={selectedRunId} onSelect={setSelectedRunId} />}
          </div>}

          {phase === "analyze" && <div className="phase-content">{selected && <AssuranceWorkbench workflow={selected} codeEvidence={codeEvidence} view="investigate" />}</div>}
          {phase === "release" && <div className="phase-content">{selected && <AssuranceWorkbench workflow={selected} codeEvidence={codeEvidence} view="release" />}</div>}
        </>}
      </div>
    </div>
  );
}
