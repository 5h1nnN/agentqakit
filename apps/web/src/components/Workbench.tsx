import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";

import { api } from "../api";
import type {
  ArtifactImportResult,
  CatalogArtifact,
  CodeEvidence,
  EvidenceBoundaryPreview,
  PreflightReport,
  WorkbenchProject,
  WorkflowEvent,
  WorkflowRun,
  WorkspaceEntry,
} from "../types";
import { GenerationWorkbench } from "./GenerationWorkbench";

function statusClass(status: string) {
  if (["COMPLETED", "PASS", "AVAILABLE"].includes(status)) return "status--good";
  if (["FAILED_SAFE", "CANCELLED", "BLOCK", "INVALID", "MISSING"].includes(status)) return "status--bad";
  if (status === "RUNNING" || status === "REVIEW" || status.startsWith("WAITING")) return "status--warn";
  return "status--neutral";
}

function shortId(value: string) {
  return value.length > 18 ? `${value.slice(0, 9)}…${value.slice(-7)}` : value;
}

function compactHash(value: string) {
  return value.replace("sha256:", "").slice(0, 12);
}

function splitValues(value: string) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

export function Workbench() {
  const [projects, setProjects] = useState<WorkbenchProject[]>([]);
  const [workflows, setWorkflows] = useState<WorkflowRun[]>([]);
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const [projectId, setProjectId] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const [newProjectId, setNewProjectId] = useState("");
  const [workspacePath, setWorkspacePath] = useState(".");
  const [tags, setTags] = useState("");
  const [manifestPath, setManifestPath] = useState("");
  const [targetPath, setTargetPath] = useState("");
  const [baselineTargetPath, setBaselineTargetPath] = useState("");
  const [candidateTargetPath, setCandidateTargetPath] = useState("");
  const [defectTargetPath, setDefectTargetPath] = useState("");
  const [fixedTargetPath, setFixedTargetPath] = useState("");
  const [testsPath, setTestsPath] = useState("");
  const [mutationsPath, setMutationsPath] = useState("");
  const [gatePath, setGatePath] = useState("");
  const [generationGoalPath, setGenerationGoalPath] = useState("");
  const [workflowName, setWorkflowName] = useState("");
  const [workflowDescription, setWorkflowDescription] = useState("");
  const [idempotencyKey, setIdempotencyKey] = useState(() => `web-${Date.now()}`);
  const [draft, setDraft] = useState(false);
  const [preflight, setPreflight] = useState<PreflightReport | null>(null);
  const [importResult, setImportResult] = useState<ArtifactImportResult | null>(null);
  const [artifacts, setArtifacts] = useState<CatalogArtifact[]>([]);
  const [artifactTotal, setArtifactTotal] = useState(0);
  const [browsePath, setBrowsePath] = useState(".");
  const [workspaceEntries, setWorkspaceEntries] = useState<WorkspaceEntry[]>([]);
  const [evidencePath, setEvidencePath] = useState("");
  const [startLine, setStartLine] = useState(1);
  const [endLine, setEndLine] = useState(40);
  const [evidenceReason, setEvidenceReason] = useState("");
  const [codePreview, setCodePreview] = useState<CodeEvidence | null>(null);
  const [selectedEvidence, setSelectedEvidence] = useState<CodeEvidence[]>([]);
  const [boundary, setBoundary] = useState<EvidenceBoundaryPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventCursor = useRef(0);

  const selected = useMemo(
    () => workflows.find((workflow) => workflow.workflow_id === selectedId) ?? null,
    [selectedId, workflows],
  );
  const selectedProject = useMemo(
    () => projects.find((project) => project.project_id === projectId) ?? null,
    [projectId, projects],
  );

  const refresh = useCallback(async () => {
    try {
      const [nextProjects, nextWorkflows] = await Promise.all([
        api.listProjects(),
        api.listWorkflows(projectId || undefined),
      ]);
      setProjects(nextProjects);
      setWorkflows(nextWorkflows);
      if (!projectId && nextProjects.length > 0) setProjectId(nextProjects[0].project_id);
      if (selectedId) {
        const timeline = await api.listWorkflowEvents(selectedId, eventCursor.current);
        eventCursor.current = timeline.revision;
        if (timeline.events.length > 0) {
          setEvents((current) => {
            const known = new Set(current.map((event) => event.event_id));
            return [...current, ...timeline.events.filter((event) => !known.has(event.event_id))];
          });
        }
      }
      setError(null);
    } catch (reason) {
      setError((reason as Error).message);
    }
  }, [projectId, selectedId]);

  useEffect(() => {
    eventCursor.current = 0;
    setEvents([]);
  }, [selectedId]);

  useEffect(() => {
    setPreflight(null);
    setImportResult(null);
    setArtifacts([]);
    setArtifactTotal(0);
    setWorkspaceEntries([]);
    setCodePreview(null);
    setSelectedEvidence([]);
    setBoundary(null);
  }, [projectId]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const createProject = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setError(null);
    const refs = Object.fromEntries(Object.entries({
      manifest: manifestPath,
      target: targetPath,
      baseline_target: baselineTargetPath,
      candidate_target: candidateTargetPath,
      defect_target: defectTargetPath,
      fixed_target: fixedTargetPath,
      tests: testsPath,
      mutations: mutationsPath,
      gate: gatePath,
      generation_goal: generationGoalPath,
    }).filter(([, value]) => value.trim()));
    try {
      const project = await api.createProject({
        projectId: newProjectId,
        name: projectName,
        description: projectDescription,
        workspacePath,
        defaultSpecRefs: refs,
        tags: splitValues(tags),
      });
      setProjectId(project.project_id);
      setNewProjectId(""); setProjectName(""); setProjectDescription(""); setTags("");
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const createWorkflow = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setError(null);
    try {
      const workflow = await api.createWorkflow({
        projectId,
        name: workflowName,
        description: workflowDescription,
        idempotencyKey,
        draft,
      });
      setSelectedId(workflow.workflow_id);
      setWorkflowName(""); setWorkflowDescription("");
      setIdempotencyKey(`web-${Date.now()}`);
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const transition = async (action: "resume" | "cancel") => {
    if (!selected) return;
    setBusy(true); setError(null);
    try {
      if (action === "resume") await api.resumeWorkflow(selected.workflow_id);
      else await api.cancelWorkflow(selected.workflow_id);
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const runPreflight = async () => {
    if (!projectId) return;
    setBusy(true); setError(null);
    try {
      setPreflight(await api.preflightProject(projectId));
      const updated = await api.getProject(projectId);
      setProjects((current) => current.map((project) => project.project_id === projectId ? updated : project));
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const refreshEvidence = async () => {
    if (!projectId) return;
    setBusy(true); setError(null);
    try {
      const [catalog, nextEvidence, nextBoundary] = await Promise.all([
        api.listArtifacts(projectId),
        api.listCodeEvidence(projectId),
        api.evidencePreview(projectId),
      ]);
      setArtifacts(catalog.artifacts);
      setArtifactTotal(catalog.total);
      setSelectedEvidence(nextEvidence);
      setBoundary(nextBoundary);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const importArtifacts = async () => {
    if (!projectId) return;
    setBusy(true); setError(null);
    try {
      setImportResult(await api.importArtifacts(projectId, [
        "artifacts", "mcp", "benchmark_results", "reference_results",
      ]));
      const catalog = await api.listArtifacts(projectId);
      setArtifacts(catalog.artifacts);
      setArtifactTotal(catalog.total);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const browse = async (path = browsePath) => {
    if (!projectId) return;
    setBusy(true); setError(null);
    try {
      setWorkspaceEntries(await api.browseWorkspace(projectId, path));
      setBrowsePath(path);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const selectionInput = { path: evidencePath, startLine, endLine, reason: evidenceReason };
  const previewCode = async (event: FormEvent) => {
    event.preventDefault();
    if (!projectId) return;
    setBusy(true); setError(null); setCodePreview(null);
    try { setCodePreview(await api.previewCodeEvidence(projectId, selectionInput)); }
    catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const addCode = async () => {
    if (!projectId || !codePreview) return;
    setBusy(true); setError(null);
    try {
      await api.addCodeEvidence(projectId, selectionInput);
      const [nextEvidence, nextBoundary] = await Promise.all([
        api.listCodeEvidence(projectId), api.evidencePreview(projectId),
      ]);
      setSelectedEvidence(nextEvidence); setBoundary(nextBoundary);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  return (
    <div className="workbench-layout">
      <section className="workbench-heading">
        <div><p className="eyebrow">Persistent local workflow</p><h1>Assurance workbench</h1><p>Configure a project, verify its readiness, index immutable evidence, and explicitly choose what AI operations may read.</p></div>
        <div className="workbench-summary"><strong>{projects.length}</strong><span>projects</span><strong>{artifactTotal}</strong><span>artifacts</span></div>
      </section>
      {error && <div className="notice notice--bad">{error}</div>}

      <div className="workbench-grid">
        <aside className="workbench-sidebar">
          <form className="panel workbench-form" onSubmit={(event) => void createProject(event)}>
            <div className="panel__header"><div><h2>New project</h2><small>All paths are workspace-relative</small></div></div>
            <div className="padded form-stack">
              <label>Project ID<input value={newProjectId} onChange={(event) => setNewProjectId(event.target.value)} placeholder="campus-ops" required /></label>
              <label>Display name<input value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="Campus Ops agent" required /></label>
              <label>Description<textarea value={projectDescription} onChange={(event) => setProjectDescription(event.target.value)} placeholder="What agent is being assured?" /></label>
              <label>Project directory<input className="mono" value={workspacePath} onChange={(event) => setWorkspacePath(event.target.value)} required /></label>
              <details><summary>Default specifications</summary><div className="form-stack compact-fields"><label>Manifest<input className="mono" value={manifestPath} onChange={(event) => setManifestPath(event.target.value)} placeholder="manifest.yaml" /></label><label>Target<input className="mono" value={targetPath} onChange={(event) => setTargetPath(event.target.value)} placeholder="target.yaml" /></label><label>Baseline target<input className="mono" value={baselineTargetPath} onChange={(event) => setBaselineTargetPath(event.target.value)} /></label><label>Candidate target<input className="mono" value={candidateTargetPath} onChange={(event) => setCandidateTargetPath(event.target.value)} /></label><label>Defect target<input className="mono" value={defectTargetPath} onChange={(event) => setDefectTargetPath(event.target.value)} /></label><label>Fixed target<input className="mono" value={fixedTargetPath} onChange={(event) => setFixedTargetPath(event.target.value)} /></label><label>Tests<input className="mono" value={testsPath} onChange={(event) => setTestsPath(event.target.value)} placeholder="tests/" /></label><label>Mutations<input className="mono" value={mutationsPath} onChange={(event) => setMutationsPath(event.target.value)} /></label><label>Gate<input className="mono" value={gatePath} onChange={(event) => setGatePath(event.target.value)} /></label><label>Generation goal<input className="mono" value={generationGoalPath} onChange={(event) => setGenerationGoalPath(event.target.value)} /></label></div></details>
              <label>Tags<input value={tags} onChange={(event) => setTags(event.target.value)} placeholder="demo, campus, long-running" /><small>Comma-separated</small></label>
              <button className="button" disabled={busy}>Create project</button>
            </div>
          </form>

          <form className="panel workbench-form" onSubmit={(event) => void createWorkflow(event)}>
            <div className="panel__header"><h2>New workflow</h2></div>
            <div className="padded form-stack"><label>Project<select value={projectId} onChange={(event) => { setProjectId(event.target.value); setSelectedId(null); }} required><option value="" disabled>Select a project</option>{projects.map((project) => <option key={project.project_id} value={project.project_id}>{project.name}</option>)}</select></label><label>Workflow name<input value={workflowName} onChange={(event) => setWorkflowName(event.target.value)} placeholder="Release candidate review" required /></label><label>Description<input value={workflowDescription} onChange={(event) => setWorkflowDescription(event.target.value)} placeholder="What should this workflow prove?" /></label><label>Idempotency key<input className="mono" value={idempotencyKey} onChange={(event) => setIdempotencyKey(event.target.value)} required /></label><label className="check-row"><input type="checkbox" checked={draft} onChange={(event) => setDraft(event.target.checked)} /><span>Keep as draft<small>Resume when the setup is ready</small></span></label><button className="button button--accent" disabled={busy || !projectId}>Create workflow</button></div>
          </form>
        </aside>

        <section className="workbench-content">
          {selectedProject && <>
            <div className="panel project-control">
              <div className="panel__header"><div><p className="eyebrow">Project control</p><h2>{selectedProject.name}</h2><small className="mono">{selectedProject.workspace_path}</small></div><div className="button-row"><button className="button button--small" disabled={busy} onClick={() => void runPreflight()}>Run preflight</button><button className="button button--small" disabled={busy} onClick={() => void importArtifacts()}>Discover artifacts</button><button className="button button--small" disabled={busy} onClick={() => void refreshEvidence()}>Refresh catalog</button></div></div>
              <div className="project-ref-grid">{Object.entries(selectedProject.default_spec_refs ?? {}).map(([kind, path]) => <span key={kind}><small>{kind.replaceAll("_", " ")}</small><code>{path}</code></span>)}{Object.keys(selectedProject.default_spec_refs ?? {}).length === 0 && <p className="muted">No default specification references were configured.</p>}</div>
              {preflight && <div className="preflight-result"><div className="readiness-verdict"><span className={`status ${preflight.executable ? "status--good" : "status--bad"}`}>{preflight.executable ? "EXECUTABLE" : "BLOCKED"}</span>{preflight.requires_approval && <span className="status status--warn">APPROVAL REQUIRED</span>}<small>{new Date(preflight.checked_at).toLocaleString()}</small></div><div className="check-list">{preflight.checks.map((check) => <article key={check.check_id}><span className={`status ${statusClass(check.status)}`}>{check.status}</span><div><strong>{check.category} / {check.check_id}</strong><p>{check.message}</p>{check.path && <code>{check.path}</code>}</div></article>)}</div></div>}
              {importResult && <div className="import-summary"><span>Scanned <strong>{importResult.scanned}</strong></span><span>Imported <strong>{importResult.imported}</strong></span><span>Unchanged <strong>{importResult.unchanged}</strong></span><span>Runs restored <strong>{importResult.runs_registered}</strong></span><span>Rejected <strong>{importResult.rejected}</strong></span>{importResult.issues.length > 0 && <details><summary>Import diagnostics</summary>{importResult.issues.map((issue) => <p key={`${issue.path}-${issue.code}`}><code>{issue.code}</code> {issue.path}: {issue.message}</p>)}</details>}</div>}
            </div>

            <div className="panel catalog-panel">
              <div className="panel__header"><div><p className="eyebrow">Verified index</p><h2>Artifact catalog</h2></div><span className="muted">showing {artifacts.length} of {artifactTotal}</span></div>
              {artifacts.length === 0 ? <p className="muted padded">Discover existing artifacts or refresh the catalog to inspect them here.</p> : <div className="table-scroll"><table><thead><tr><th>Type</th><th>Source</th><th>Path</th><th>Created</th><th>Hash</th><th>Status</th></tr></thead><tbody>{artifacts.map((artifact) => <tr key={artifact.artifact_id}><td>{artifact.artifact_type}</td><td>{artifact.source}</td><td className="mono">{artifact.path}</td><td>{new Date(artifact.created_at).toLocaleString()}</td><td><code title={artifact.checksum}>{compactHash(artifact.checksum)}</code></td><td><span className={`status ${statusClass(artifact.availability)}`}>{artifact.availability}</span></td></tr>)}</tbody></table></div>}
            </div>

            <div className="panel evidence-picker">
              <div className="panel__header"><div><p className="eyebrow">Explicit evidence boundary</p><h2>Code evidence</h2></div><span className="muted">{selectedEvidence.length}/10 selected</span></div>
              <div className="evidence-grid">
                <section className="workspace-browser"><h3>1. Browse safe workspace</h3><div className="inline-form"><input className="mono" value={browsePath} onChange={(event) => setBrowsePath(event.target.value)} /><button className="button button--small" disabled={busy} onClick={() => void browse()}>Browse</button></div><div className="workspace-entries">{workspaceEntries.map((entry) => <button key={entry.path} onClick={() => { if (entry.kind === "directory") void browse(entry.path); else { setEvidencePath(entry.path); setCodePreview(null); } }}><span>{entry.kind === "directory" ? "DIR" : "FILE"}</span><code>{entry.path}</code><small>{entry.size_bytes === null ? "" : `${entry.size_bytes} B`}</small></button>)}</div></section>
                <form className="code-selection form-stack" onSubmit={(event) => void previewCode(event)}><h3>2. Select a bounded excerpt</h3><label>File<input className="mono" value={evidencePath} onChange={(event) => { setEvidencePath(event.target.value); setCodePreview(null); }} placeholder="src/agent.py" required /></label><div className="line-range"><label>Start line<input type="number" min="1" value={startLine} onChange={(event) => { setStartLine(Number(event.target.value)); setCodePreview(null); }} required /></label><label>End line<input type="number" min="1" value={endLine} onChange={(event) => { setEndLine(Number(event.target.value)); setCodePreview(null); }} required /></label></div><label>Selection reason<textarea value={evidenceReason} onChange={(event) => { setEvidenceReason(event.target.value); setCodePreview(null); }} placeholder="Why is this excerpt relevant to test generation or diagnosis?" required /></label><button className="button" disabled={busy || !projectId}>Preview redacted excerpt</button></form>
              </div>
              {codePreview && <section className="code-preview"><div><strong>{codePreview.path}:{codePreview.startLine}-{codePreview.endLine}</strong><code>{codePreview.contentHash}</code></div><pre>{codePreview.content}</pre><button className="button button--accent" disabled={busy} onClick={() => void addCode()}>Add to CodeEvidence</button><small>Previewing does not grant AI access. Only this explicit action records the selection.</small></section>}
              {selectedEvidence.length > 0 && <div className="selected-evidence"><h3>Selected evidence</h3>{selectedEvidence.map((item) => <article key={`${item.path}-${item.startLine}-${item.endLine}`}><div><strong>{item.path}:{item.startLine}-{item.endLine}</strong><p>{item.reason}</p></div><code title={item.contentHash}>{compactHash(item.contentHash)}</code></article>)}</div>}
            </div>

            <div className="panel boundary-panel">
              <div className="panel__header"><div><p className="eyebrow">Before an AI call</p><h2>Evidence boundary preview</h2></div>{boundary && <code>{compactHash(boundary.evidence_hash)}</code>}</div>
              {!boundary ? <p className="muted padded">Refresh the catalog to calculate the current evidence boundary.</p> : <div className="boundary-grid"><section><h3>Included</h3><p><strong>Manifest sections</strong> {boundary.manifest_sections.join(", ") || "none"}</p><p><strong>Evidence types</strong> {boundary.evidence_types.join(", ") || "none"}</p><p><strong>Code excerpts</strong> {boundary.code_evidence.length}</p></section><section><h3>Excluded</h3><ul>{boundary.excluded.map((item) => <li key={item}>{item}</li>)}</ul></section><section><h3>Model budget</h3><p>{boundary.provider ?? "No provider"} / {boundary.model ?? "No model"}</p><p>{boundary.max_input_tokens.toLocaleString()} input · {boundary.max_output_tokens.toLocaleString()} output tokens</p><p>{boundary.timeout_seconds}s timeout · {boundary.max_cost_usd ? `$${boundary.max_cost_usd} maximum` : "no cost ceiling configured"}</p></section></div>}
            </div>
          </>}

          <div className="panel workflow-list">
            <div className="panel__header"><div><h2>Workflows</h2><small>{projectId || "All projects"}</small></div><button className="button button--small" onClick={() => void refresh()} disabled={busy}>Refresh</button></div>
            {workflows.length === 0 ? <div className="empty-state"><div className="empty-state__mark">WF</div><h2>No workflows yet</h2><p>Create a project and its first durable workflow.</p></div> : <div className="workflow-items">{workflows.map((workflow) => <button key={workflow.workflow_id} className={workflow.workflow_id === selectedId ? "workflow-item is-selected" : "workflow-item"} onClick={() => setSelectedId(workflow.workflow_id)}><span><strong>{workflow.name}</strong><small className="mono">{shortId(workflow.workflow_id)} · rev {workflow.revision}</small></span><span className={`status ${statusClass(workflow.status)}`}>{workflow.status}</span></button>)}</div>}
          </div>

          {selected && <div className="panel workflow-detail">
            <div className="panel__header"><div><p className="eyebrow">Workflow detail</p><h2>{selected.name}</h2></div><div className="button-row"><button className="button button--small" disabled={busy || !["DRAFT", "INTERRUPTED", "WAITING_EXTERNAL_REPAIR"].includes(selected.status)} onClick={() => void transition("resume")}>Resume</button><button className="button button--small" disabled={busy || ["COMPLETED", "FAILED_SAFE", "CANCELLED"].includes(selected.status)} onClick={() => void transition("cancel")}>Cancel</button></div></div>
            <div className="workflow-meta"><span>Status<strong className={`status ${statusClass(selected.status)}`}>{selected.status}</strong></span><span>Revision<strong>{selected.revision}</strong></span><span>Steps<strong>{selected.steps.length}</strong></span><span>Updated<strong>{new Date(selected.updated_at).toLocaleString()}</strong></span></div>
            <div className="workflow-hash"><span>Input snapshot</span><code>{selected.input_hash}</code></div>
            {selected.error && <div className="notice notice--bad"><strong>{selected.error.code}</strong> — {selected.error.message}</div>}
            {selected.steps.length > 0 && <div className="step-progress"><h3>Step attempts</h3>{selected.steps.map((step) => <article key={step.step_id}><span><strong>{step.kind.replaceAll("_", " ")}</strong><small className="mono">attempt {step.attempt} · {shortId(step.step_id)} · input {shortId(step.input_hash)}</small></span><span className={`status ${statusClass(step.status)}`}>{step.status}</span>{step.error && <p>{step.error.code}: {step.error.message}</p>}</article>)}</div>}
            {selectedProject && <GenerationWorkbench project={selectedProject} workflow={selected} codeEvidence={selectedEvidence} />}
            <div className="timeline"><h3>Audit timeline</h3>{events.map((item) => <article key={item.event_id}><span className="timeline__revision">{item.revision}</span><div><strong>{item.event_type.replaceAll("_", " ")}</strong><p>{item.message}</p><small>{new Date(item.occurred_at).toLocaleString()}</small></div></article>)}</div>
          </div>}
        </section>
      </div>
    </div>
  );
}
