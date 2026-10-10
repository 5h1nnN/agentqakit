import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";

import { api } from "../api";
import { duration, milliseconds, percent, shortId } from "../lib";
import type { StoredFinding, StoredRunSummary, TestResult, TraceRecord } from "../types";
import { StatusBadge } from "./StatusBadge";
import { TracePanel } from "./TracePanel";

export function RunDetail({ run }: { run: StoredRunSummary }) {
  const [results, setResults] = useState<TestResult[]>([]);
  const [findings, setFindings] = useState<StoredFinding[]>([]);
  const [trace, setTrace] = useState<TraceRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setTrace(null);
    setError(null);
    Promise.all([api.getResults(run.run_id), api.getFindings(run.run_id)])
      .then(([nextResults, nextFindings]) => {
        setResults(nextResults);
        setFindings(nextFindings);
      })
      .catch((reason: Error) => setError(reason.message));
  }, [run.run_id]);

  const mutations = useMemo(
    () => results.flatMap((result) => result.mutation_applications.map((item) => ({ ...item, testCaseId: result.test_case_id }))),
    [results],
  );
  const judgeEvaluations = useMemo(
    () => results.flatMap((result) => result.evaluations
      .filter((evaluation) => evaluation.evaluator_type === "llm_judge")
      .map((evaluation) => ({ ...evaluation, testCaseId: result.test_case_id, trialId: result.trial_id }))),
    [results],
  );

  const showTrace = async (traceId: string) => {
    try {
      setTrace(await api.getTrace(traceId));
    } catch (reason) {
      setError((reason as Error).message);
    }
  };

  return (
    <div className="detail-stack">
      {error && <div className="notice notice--bad">{error}</div>}
      <section className="run-hero panel">
        <div>
          <p className="eyebrow">Run summary</p>
          <h2 className="mono">{run.run_id}</h2>
          <div className="run-meta"><StatusBadge status={run.status} /><span>{new Date(run.started_at).toLocaleString()}</span><span>Project {run.project_id}</span></div>
        </div>
        <div className="score-ring" style={{ "--score": `${run.summary.pass_rate * 100}%` } as CSSProperties}>
          <strong>{percent(run.summary.pass_rate)}</strong><span>pass rate</span>
        </div>
      </section>
      <section className="metric-grid">
        <article className="metric"><span>Passed</span><strong>{run.summary.passed}</strong><small>of {run.summary.total} cases</small></article>
        <article className="metric"><span>Mutation score</span><strong>{percent(run.mutation_score)}</strong><small>fault sensitivity</small></article>
        <article className="metric"><span>Findings</span><strong>{run.finding_count}</strong><small>reviewable violations</small></article>
        <article className="metric"><span>Consistency</span><strong>{run.summary.inconsistent_cases}</strong><small>inconsistent cases</small></article>
      </section>

      <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Trial evidence</p><h3>Test results</h3></div><span className="muted">{results.length} trials</span></div>
        <div className="table-scroll"><table><thead><tr><th>Test case</th><th>Trial</th><th>Outcome</th><th>Target</th><th>Latency</th><th>Trace</th><th>Evidence</th></tr></thead>
          <tbody>{results.map((result) => <tr key={result.trial_id}><td>{result.test_case_id}</td><td className="mono">{shortId(result.trial_id)}</td><td><StatusBadge status={result.passed ? "PASS" : "FAIL"} /></td><td><StatusBadge status={result.target_result.status} /></td><td>{duration(milliseconds(result.started_at, result.ended_at))}</td><td><StatusBadge status={result.trace_status} /></td><td><button className="button button--small" onClick={() => void showTrace(result.trace_id)}>Inspect</button></td></tr>)}</tbody>
        </table></div>
      </section>

      <TracePanel trace={trace} />

      {judgeEvaluations.length > 0 && <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Semantic evaluation</p><h3>DeepSeek AI judge</h3></div><span className="muted">{judgeEvaluations.length} decisions</span></div>
        <div className="judge-list">{judgeEvaluations.map((evaluation) => <article className="judge-card" key={`${evaluation.trialId}-${evaluation.evaluator_type}`}><StatusBadge status={evaluation.status} /><div><strong>{evaluation.testCaseId}</strong><p>{evaluation.message ?? "No rationale was returned."}</p></div><span className="judge-score">{evaluation.score === null ? "N/A" : percent(evaluation.score)}</span></article>)}</div>
      </section>}

      <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Failure analysis</p><h3>Findings</h3></div></div>
        {findings.length === 0 ? <p className="muted padded">No findings were emitted for this run.</p> : <div className="finding-list">{findings.map((finding) => <article className="finding" key={finding.finding_id}><div><StatusBadge status="FAIL">{finding.type}</StatusBadge><h4>{finding.message}</h4><p className="mono muted">{shortId(finding.finding_id)}</p></div><details><summary>Details</summary><pre>{JSON.stringify(finding.details, null, 2)}</pre></details><button className="button button--ghost" onClick={() => void api.updateReview(finding.finding_id, "confirmed")}>Confirm</button></article>)}</div>}
      </section>

      <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Fault coverage</p><h3>Mutation matrix</h3></div></div>
        {mutations.length === 0 ? <p className="muted padded">No mutation applications were recorded.</p> : <div className="table-scroll"><table><thead><tr><th>Test</th><th>Mutation</th><th>Operator</th><th>Call</th><th>Triggered</th></tr></thead><tbody>{mutations.map((item, index) => <tr key={`${item.mutation_id}-${index}`}><td>{item.testCaseId}</td><td>{item.mutation_id}</td><td>{item.operation_type}</td><td>{item.call_index}</td><td><StatusBadge status={item.triggered ? "PASS" : "NEEDS_REVIEW"}>{item.triggered ? "YES" : "NO"}</StatusBadge></td></tr>)}</tbody></table></div>}
      </section>
    </div>
  );
}
