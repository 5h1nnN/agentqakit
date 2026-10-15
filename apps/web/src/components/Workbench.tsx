import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";

import { api } from "../api";
import type {
  WorkbenchProject,
  WorkflowEvent,
  WorkflowRun,
} from "../types";

function statusClass(status: string) {
  if (status === "COMPLETED") return "status--good";
  if (status === "FAILED_SAFE" || status === "CANCELLED") return "status--bad";
  if (status === "RUNNING" || status.startsWith("WAITING")) return "status--warn";
  return "status--neutral";
}

function shortId(value: string) {
  return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value;
}

export function Workbench() {
  const [projects, setProjects] = useState<WorkbenchProject[]>([]);
  const [workflows, setWorkflows] = useState<WorkflowRun[]>([]);
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const [projectId, setProjectId] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [projectName, setProjectName] = useState("");
  const [newProjectId, setNewProjectId] = useState("");
  const [workflowName, setWorkflowName] = useState("");
  const [workflowDescription, setWorkflowDescription] = useState("");
  const [idempotencyKey, setIdempotencyKey] = useState(() => `web-${Date.now()}`);
  const [draft, setDraft] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventCursor = useRef(0);

  const selected = useMemo(
    () => workflows.find((workflow) => workflow.workflow_id === selectedId) ?? null,
    [selectedId, workflows],
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
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const createProject = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setError(null);
    try {
      const project = await api.createProject({ projectId: newProjectId, name: projectName });
      setProjectId(project.project_id);
      setNewProjectId(""); setProjectName("");
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

  return (
    <div className="workbench-layout">
      <section className="workbench-heading">
        <div><p className="eyebrow">Persistent local workflow</p><h1>Assurance workbench</h1><p>Create a durable workflow, inspect every state transition, and safely resume interrupted work.</p></div>
        <div className="workbench-summary"><strong>{projects.length}</strong><span>projects</span><strong>{workflows.length}</strong><span>workflows</span></div>
      </section>
      {error && <div className="notice notice--bad">{error}</div>}

      <div className="workbench-grid">
        <aside className="workbench-sidebar">
          <form className="panel workbench-form" onSubmit={(event) => void createProject(event)}>
            <div className="panel__header"><h2>New project</h2></div>
            <div className="padded form-stack"><label>Project ID<input value={newProjectId} onChange={(event) => setNewProjectId(event.target.value)} placeholder="my-agent" required /></label><label>Display name<input value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="My agent" required /></label><button className="button" disabled={busy}>Create project</button></div>
          </form>

          <form className="panel workbench-form" onSubmit={(event) => void createWorkflow(event)}>
            <div className="panel__header"><h2>New workflow</h2></div>
            <div className="padded form-stack"><label>Project<select value={projectId} onChange={(event) => { setProjectId(event.target.value); setSelectedId(null); }} required><option value="" disabled>Select a project</option>{projects.map((project) => <option key={project.project_id} value={project.project_id}>{project.name}</option>)}</select></label><label>Workflow name<input value={workflowName} onChange={(event) => setWorkflowName(event.target.value)} placeholder="Release candidate review" required /></label><label>Description<input value={workflowDescription} onChange={(event) => setWorkflowDescription(event.target.value)} placeholder="What should this workflow prove?" /></label><label>Idempotency key<input className="mono" value={idempotencyKey} onChange={(event) => setIdempotencyKey(event.target.value)} required /></label><label className="check-row"><input type="checkbox" checked={draft} onChange={(event) => setDraft(event.target.checked)} /><span>Keep as draft<small>Resume when the setup is ready</small></span></label><button className="button button--accent" disabled={busy || !projectId}>Create workflow</button></div>
          </form>
        </aside>

        <section className="workbench-content">
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
            <div className="timeline"><h3>Audit timeline</h3>{events.map((item) => <article key={item.event_id}><span className="timeline__revision">{item.revision}</span><div><strong>{item.event_type.replaceAll("_", " ")}</strong><p>{item.message}</p><small>{new Date(item.occurred_at).toLocaleString()}</small></div></article>)}</div>
          </div>}
        </section>
      </div>
    </div>
  );
}
