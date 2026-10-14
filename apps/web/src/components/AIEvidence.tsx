import { useEffect, useState } from "react";

import { api } from "../api";
import { percent, shortId } from "../lib";
import type { AIEvidenceResponse } from "../types";
import { StatusBadge } from "./StatusBadge";

export function AIEvidence() {
  const [evidence, setEvidence] = useState<AIEvidenceResponse>({ generations: [], analyses: [] });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.aiEvidence().then(setEvidence).catch((reason: Error) => setError(reason.message));
  }, []);

  return (
    <div className="quality-layout">
      {error && <div className="notice notice--bad">{error}</div>}
      <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Bounded generation evidence</p><h2>AI test generation</h2></div><span className="muted">{evidence.generations.length} reports</span></div>
        {evidence.generations.length === 0 ? <p className="muted padded">No verified generation reports were found.</p> : <div className="finding-list">{evidence.generations.map((report) => {
          const reviewRequired = report.cases.filter((item) => item.review_status === "REVIEW_REQUIRED").length;
          const totalTokens = report.llm_calls.reduce((total, call) => total + (call.token_usage.total_tokens ?? 0), 0);
          return <article className="finding" key={report.generation_id}><div><StatusBadge status={reviewRequired ? "NEEDS_REVIEW" : "PASS"}>{reviewRequired ? `${reviewRequired} REVIEW` : "READY"}</StatusBadge><h4>{shortId(report.generation_id)} · {report.cases.length} tests</h4><p className="mono muted">{report.path}</p></div><details><summary>Evidence and approvals</summary><p>Manifest <code>{report.manifest_hash}</code></p><p>Code contexts {report.context_hashes.length} · coverage gaps {report.coverage_gaps} · tokens {totalTokens}</p><div className="table-scroll"><table><thead><tr><th>Case</th><th>Risk</th><th>Approval</th></tr></thead><tbody>{report.cases.map((item) => <tr key={item.case_id}><td>{item.case_id}</td><td>{item.risk}</td><td><StatusBadge status={item.review_status}>{item.review_status}</StatusBadge></td></tr>)}</tbody></table></div></details></article>;
        })}</div>}
      </section>

      <section className="panel table-panel">
        <div className="panel__header"><div><p className="eyebrow">Evidence-citing diagnosis</p><h2>AI failure analysis</h2></div><span className="muted">{evidence.analyses.length} reports</span></div>
        {evidence.analyses.length === 0 ? <p className="muted padded">No verified analysis reports were found.</p> : <div className="finding-list">{evidence.analyses.map((report) => <article className="finding" key={report.analysis_id}><div><StatusBadge status={report.missing_evidence.length ? "NEEDS_REVIEW" : "PASS"}>{report.root_causes.length} CAUSES</StatusBadge><h4>{report.test_case_id}</h4><p className="mono muted">Run {shortId(report.run_id)} · {report.llm_call.model}</p></div><details><summary>Ranked hypotheses</summary>{report.root_causes.map((cause, index) => <div key={cause.id}><strong>#{index + 1} {cause.id} · {percent(cause.confidence)}</strong><p>{cause.hypothesis}</p><small className="mono muted">Evidence {cause.evidence_ids.join(", ")}</small></div>)}</details></article>)}</div>}
      </section>
    </div>
  );
}
