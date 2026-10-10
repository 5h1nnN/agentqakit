import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";

import { api } from "./api";
import { QualityLab } from "./components/QualityLab";
import { RunDetail } from "./components/RunDetail";
import { RunList } from "./components/RunList";
import type { RunJob, StoredRunSummary } from "./types";

type View = "runs" | "quality";

export default function App() {
  const [view, setView] = useState<View>("runs");
  const [runs, setRuns] = useState<StoredRunSummary[]>([]);
  const [jobs, setJobs] = useState<RunJob[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);
  const [projectId, setProjectId] = useState("default");
  const [testsPath, setTestsPath] = useState("examples/specs/ci-test.yaml");
  const [targetPath, setTargetPath] = useState("examples/specs/ci-candidate-target.yaml");
  const [mutationsPath, setMutationsPath] = useState("");

  const refresh = useCallback(async () => {
    try {
      const data = await api.listRuns(projectId);
      setRuns(data.runs);
      setJobs(data.jobs);
      setError(null);
    } catch (reason) {
      setError((reason as Error).message);
    }
  }, [projectId]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const selected = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId) ?? null,
    [runs, selectedRunId],
  );

  const launch = async (event: FormEvent) => {
    event.preventDefault(); setLaunching(true); setError(null);
    try {
      const job = await api.createRun({ projectId, testsPath, targetPath: targetPath || undefined, mutationsPath: mutationsPath || undefined });
      setJobs((current) => [job, ...current]);
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setLaunching(false); }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => { setView("runs"); setSelectedRunId(null); }}><span className="brand__mark">AQ</span><span><strong>AgentQAKit</strong><small>Evidence console</small></span></button>
        <nav aria-label="Primary"><button className={view === "runs" ? "active" : ""} onClick={() => setView("runs")}>Runs</button><button className={view === "quality" ? "active" : ""} onClick={() => setView("quality")}>Compare &amp; gate</button></nav>
        <div className="live-indicator"><span /> Local</div>
      </header>

      <main>
        {error && <div className="notice notice--bad global-notice">{error}</div>}
        {view === "runs" && !selected && <>
          <section className="intro"><div><p className="eyebrow">Local-first agent assurance</p><h1>Turn every run into<br /><em>reviewable evidence.</em></h1><p>Execute deterministic tests, inspect traces, and carry verified findings into release decisions.</p></div>
            <form className="launch-card" onSubmit={(event) => void launch(event)}><div className="launch-card__title"><span>+</span><div><strong>Launch a run</strong><small>Workspace-relative specs only</small></div></div><label>Project<input value={projectId} onChange={(event) => setProjectId(event.target.value)} required /></label><label>Tests path<input value={testsPath} onChange={(event) => setTestsPath(event.target.value)} required /></label><label>Target path<input value={targetPath} onChange={(event) => setTargetPath(event.target.value)} /></label><label>Mutations path <small>optional</small><input value={mutationsPath} onChange={(event) => setMutationsPath(event.target.value)} /></label><button className="button button--accent" disabled={launching}>{launching ? "Submitting…" : "Run test suite"}</button></form>
          </section>
          <RunList runs={runs} jobs={jobs} selectedRunId={selectedRunId} onSelect={setSelectedRunId} />
        </>}
        {view === "runs" && selected && <><button className="back-button" onClick={() => setSelectedRunId(null)}>← All runs</button><RunDetail run={selected} /></>}
        {view === "quality" && <QualityLab runs={runs} />}
      </main>
      <footer><span>AgentQAKit v0.1 alpha</span><span>Artifacts remain the source of truth</span></footer>
    </div>
  );
}
