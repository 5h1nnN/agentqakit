import { duration, percent, shortId } from "../lib";
import type { RunJob, StoredRunSummary } from "../types";
import { StatusBadge } from "./StatusBadge";

interface Props {
  runs: StoredRunSummary[];
  jobs: RunJob[];
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
}

export function RunList({ runs, jobs, selectedRunId, onSelect }: Props) {
  if (!runs.length && !jobs.length) {
    return (
      <section className="empty-state panel">
        <div className="empty-state__mark">AQ</div>
        <h2>No runs yet</h2>
        <p>Launch a workspace test above. Results, traces, and findings will appear here.</p>
      </section>
    );
  }
  return (
    <section className="panel table-panel" aria-labelledby="runs-title">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Evidence ledger</p>
          <h2 id="runs-title">Runs</h2>
        </div>
        <span className="muted">{runs.length + jobs.filter((job) => !runs.some((run) => run.run_id === job.runId)).length} records</span>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Pass rate</th>
              <th>Mutation</th>
              <th>P95 latency</th>
              <th>Findings</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody>
            {jobs
              .filter((job) => !runs.some((run) => run.run_id === job.runId))
              .map((job) => (
                <tr key={job.runId}>
                  <td><span className="mono">{shortId(job.runId)}</span></td>
                  <td><StatusBadge status={job.status} /></td>
                  <td colSpan={4} className="muted">Waiting for finalized evidence</td>
                  <td>{new Date(job.submittedAt).toLocaleTimeString()}</td>
                </tr>
              ))}
            {runs.map((run) => {
              const caseP95 = run.summary.cases
                .map((item) => item.latency_ms.p95)
                .filter((value): value is number => value !== null);
              const p95 = caseP95.length ? Math.max(...caseP95) : null;
              return (
                <tr
                  key={run.run_id}
                  className={selectedRunId === run.run_id ? "is-selected" : undefined}
                  onClick={() => onSelect(run.run_id)}
                  tabIndex={0}
                  onKeyDown={(event) => event.key === "Enter" && onSelect(run.run_id)}
                >
                  <td><button className="link-button mono">{shortId(run.run_id)}</button></td>
                  <td><StatusBadge status={run.status} /></td>
                  <td>{percent(run.summary.pass_rate)}</td>
                  <td>{percent(run.mutation_score)}</td>
                  <td>{duration(p95)}</td>
                  <td>{run.finding_count}</td>
                  <td>{new Date(run.started_at).toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
