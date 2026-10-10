import { useEffect, useState } from "react";

import { api } from "../api";
import { percent, shortId } from "../lib";
import type { ComparisonReport, GateResult, StoredRunSummary } from "../types";
import { StatusBadge } from "./StatusBadge";

export function QualityLab({ runs }: { runs: StoredRunSummary[] }) {
  const [baseline, setBaseline] = useState(runs[1]?.run_id ?? runs[0]?.run_id ?? "");
  const [candidate, setCandidate] = useState(runs[0]?.run_id ?? "");
  const [gatePath, setGatePath] = useState("examples/specs/quality-gate.yaml");
  const [comparison, setComparison] = useState<ComparisonReport | null>(null);
  const [gate, setGate] = useState<GateResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!candidate && runs[0]) setCandidate(runs[0].run_id);
    if (!baseline && runs.length > 0) setBaseline(runs[1]?.run_id ?? runs[0].run_id);
  }, [baseline, candidate, runs]);

  const runComparison = async () => {
    setBusy(true); setError(null);
    try { setComparison(await api.compare(baseline, candidate)); }
    catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  const runGate = async () => {
    setBusy(true); setError(null);
    try { setGate(await api.gate(gatePath, candidate, baseline || undefined)); }
    catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  return (
    <div className="quality-layout">
      <section className="panel quality-controls">
        <div className="panel__header"><div><p className="eyebrow">Release decision</p><h2>Comparison &amp; gate</h2></div></div>
        {error && <div className="notice notice--bad">{error}</div>}
        <div className="form-grid">
          <label>Baseline<select value={baseline} onChange={(event) => setBaseline(event.target.value)}><option value="">Select run</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{shortId(run.run_id)} · {percent(run.summary.pass_rate)}</option>)}</select></label>
          <label>Candidate<select value={candidate} onChange={(event) => setCandidate(event.target.value)}><option value="">Select run</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{shortId(run.run_id)} · {percent(run.summary.pass_rate)}</option>)}</select></label>
          <label className="span-2">QualityGate path<input value={gatePath} onChange={(event) => setGatePath(event.target.value)} /></label>
        </div>
        <div className="button-row"><button className="button" disabled={busy || !baseline || !candidate} onClick={() => void runComparison()}>Compare runs</button><button className="button button--accent" disabled={busy || !candidate || !gatePath} onClick={() => void runGate()}>Evaluate gate</button></div>
      </section>

      {comparison && <section className="panel">
        <div className="panel__header"><div><p className="eyebrow">Aligned by test_case_id</p><h3>Regression report</h3></div><strong>{percent(comparison.candidate_effective_pass_rate)} effective</strong></div>
        <div className="metric-grid metric-grid--compact"><article className="metric"><span>Regressions</span><strong>{comparison.counts.new_regressions}</strong></article><article className="metric"><span>Fixed</span><strong>{comparison.counts.fixed}</strong></article><article className="metric"><span>Removed</span><strong>{comparison.counts.removed}</strong></article><article className="metric"><span>Persistent</span><strong>{comparison.counts.persistent_failures}</strong></article></div>
        <div className="case-grid">{comparison.cases.filter((item) => item.classification !== "STABLE").map((item) => <article key={item.test_case_id} className="case-card"><StatusBadge status={item.classification} /><strong>{item.test_case_id}</strong><span className="muted">Risk {item.risk ?? "unlabelled"}</span></article>)}</div>
      </section>}

      {gate && <section className={`panel gate-result ${gate.passed ? "gate-result--pass" : "gate-result--fail"}`}>
        <div className="gate-verdict"><span>{gate.passed ? "PASS" : "FAIL"}</span><div><p className="eyebrow">Quality gate</p><h3>{gate.passed ? "Candidate is releasable" : "Candidate is blocked"}</h3></div></div>
        <div className="rule-list">{gate.rules.map((rule) => <article key={rule.name}><StatusBadge status={rule.passed ? "PASS" : "FAIL"} /><div><strong>{rule.name}</strong><p>{rule.message}</p></div><code>{JSON.stringify(rule.actual)} / {JSON.stringify(rule.expected)}</code></article>)}</div>
      </section>}
    </div>
  );
}
