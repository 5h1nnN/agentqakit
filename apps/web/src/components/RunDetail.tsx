import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";

import { api } from "../api";
import { duration, milliseconds, percent, shortId } from "../lib";
import type { StoredFinding, StoredRunDetail, StoredRunSummary, TestResult, TraceRecord } from "../types";
import { StatusBadge } from "./StatusBadge";
import { TracePanel } from "./TracePanel";

export function RunDetail({ run }: { run: StoredRunSummary }) {
  const [results, setResults] = useState<TestResult[]>([]);
  const [findings, setFindings] = useState<StoredFinding[]>([]);
  const [trace, setTrace] = useState<TraceRecord | null>(null);
  const [stored, setStored] = useState<StoredRunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setTrace(null);
    setError(null);
    Promise.all([api.getResults(run.run_id), api.getFindings(run.run_id), api.getRun(run.run_id)])
      .then(([nextResults, nextFindings, nextStored]) => {
        setResults(nextResults);
        setFindings(nextFindings);
        setStored(nextStored);
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
  const deterministicEvaluations = useMemo(
    () => results.flatMap((result) => result.evaluations.filter((evaluation) => evaluation.evaluator_type !== "llm_judge").map((evaluation) => ({ ...evaluation, testCaseId: result.test_case_id, trialId: result.trial_id }))),
    [results],
  );
  const replayEvidence = useMemo(() => results.filter((result) => result.replay !== null), [results]);

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

      {deterministicEvaluations.length > 0 && <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Deterministic evaluation</p><h3>Evaluator decisions</h3></div><span className="muted">{deterministicEvaluations.length} decisions</span></div>
        <div className="table-scroll"><table><thead><tr><th>Test</th><th>Evaluator</th><th>Status</th><th>Score</th><th>Evidence</th></tr></thead><tbody>{deterministicEvaluations.map((evaluation) => <tr key={`${evaluation.trialId}-${evaluation.evaluator_type}`}><td>{evaluation.testCaseId}</td><td>{evaluation.evaluator_type}</td><td><StatusBadge status={evaluation.status} /></td><td>{evaluation.score === null ? "N/A" : percent(evaluation.score)}</td><td>{evaluation.message ?? "No additional message"}</td></tr>)}</tbody></table></div>
      </section>}

      {judgeEvaluations.length > 0 && <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Semantic evaluation</p><h3>DeepSeek AI judge</h3></div><span className="muted">{judgeEvaluations.length} decisions</span></div>
        <div className="judge-list">{judgeEvaluations.map((evaluation) => <article className="judge-card" key={`${evaluation.trialId}-${evaluation.evaluator_type}`}><StatusBadge status={evaluation.status} /><div><strong>{evaluation.testCaseId}</strong><p>{evaluation.message ?? "No rationale was returned."}</p></div><span className="judge-score">{evaluation.score === null ? "N/A" : percent(evaluation.score)}</span></article>)}</div>
      </section>}

      {replayEvidence.length > 0 && <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Deterministic transport evidence</p><h3>Record / replay accounting</h3></div></div>
        <div className="table-scroll"><table><thead><tr><th>Test</th><th>Mode</th><th>Recorded</th><th>Hits</th><th>Misses</th><th>Live fallback</th><th>Cassette</th></tr></thead><tbody>{replayEvidence.map((result) => <tr key={result.trial_id}><td>{result.test_case_id}</td><td>{result.replay?.mode}</td><td>{result.replay?.recorded}</td><td>{result.replay?.hits}</td><td>{result.replay?.misses}</td><td>{result.replay?.live_fallbacks}</td><td><code>{result.cassette_ref ?? "—"}</code></td></tr>)}</tbody></table></div>
      </section>}

      <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Failure analysis</p><h3>Findings</h3></div></div>
        {findings.length === 0 ? <p className="muted padded">No findings were emitted for this run.</p> : <div className="finding-list">{findings.map((finding) => { const trial = results.find((result) => result.trial_id === finding.trial_id); const filename = trial?.failure_bundle_ref?.split("/").at(-1); return <article className="finding" key={finding.finding_id}><div><StatusBadge status="FAIL">{finding.type}</StatusBadge><h4>{finding.message}</h4><p className="mono muted">{shortId(finding.finding_id)}</p>{filename && <a className="button button--small" href={`/api/runs/${encodeURIComponent(run.run_id)}/failure-bundles/${encodeURIComponent(filename)}`}>Download verified reproduction</a>}</div><details><summary>Details</summary><pre>{JSON.stringify(finding.details, null, 2)}</pre></details><button className="button button--ghost" onClick={() => void api.updateReview(finding.finding_id, "confirmed")}>Confirm</button></article>; })}</div>}
      </section>

      <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Fault coverage</p><h3>Mutation matrix</h3></div></div>
        {stored?.run.mutation_score ? <><div className="mutation-denominator"><span>Killed <strong>{stored.run.mutation_score.overall.killed}</strong></span><span>Survived <strong>{stored.run.mutation_score.overall.survived}</strong></span><span>Not reached <strong>{stored.run.mutation_score.overall.not_reached}</strong></span><span>Invalid <strong>{stored.run.mutation_score.overall.invalid}</strong></span><span>Denominator <strong>{stored.run.mutation_score.overall.killed + stored.run.mutation_score.overall.survived}</strong></span></div><div className="table-scroll"><table><thead><tr><th>Mutation</th><th>Operator</th><th>Target kind</th><th>Status</th><th>Triggered</th><th>Failed cases</th><th>Reason</th></tr></thead><tbody>{stored.run.mutation_score.outcomes.map((item) => <tr key={item.mutation_id}><td>{item.mutation_id}</td><td>{item.operation_type}</td><td>{item.target_kind}</td><td><StatusBadge status={item.status} /></td><td>{item.triggered_count}</td><td>{item.failed_test_case_ids.join(", ") || "—"}</td><td>{item.reason ?? "—"}</td></tr>)}</tbody></table></div></> : mutations.length === 0 ? <p className="muted padded">No mutation applications were recorded.</p> : <div className="table-scroll"><table><thead><tr><th>Test</th><th>Mutation</th><th>Operator</th><th>Call</th><th>Triggered</th></tr></thead><tbody>{mutations.map((item, index) => <tr key={`${item.mutation_id}-${index}`}><td>{item.testCaseId}</td><td>{item.mutation_id}</td><td>{item.operation_type}</td><td>{item.call_index}</td><td><StatusBadge status={item.triggered ? "PASS" : "NEEDS_REVIEW"}>{item.triggered ? "YES" : "NO"}</StatusBadge></td></tr>)}</tbody></table></div>}
      </section>
    </div>
  );
}
