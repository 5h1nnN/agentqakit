import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";

import { api } from "../api";
import type {
  CodeEvidence,
  GeneratedCandidate,
  GenerationSession,
  GenerationSessionDetail,
  WorkbenchProject,
  WorkflowRun,
} from "../types";

const CATEGORIES = ["normal", "boundary", "fault", "security", "efficiency", "hitl"];
const RISKS = ["low", "medium", "high", "critical"];

function splitValues(value: string) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function shortHash(value: string | null) {
  return value ? value.replace("sha256:", "").slice(0, 12) : "—";
}

function toggle(values: string[], value: string) {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

interface Props {
  project: WorkbenchProject;
  workflow: WorkflowRun;
  codeEvidence: CodeEvidence[];
}

export function GenerationWorkbench({ project, workflow, codeEvidence }: Props) {
  const [sessions, setSessions] = useState<GenerationSession[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [detail, setDetail] = useState<GenerationSessionDetail | null>(null);
  const [goalMode, setGoalMode] = useState<"saved" | "new">(
    project.default_spec_refs.generation_goal ? "saved" : "new",
  );
  const [goalPath, setGoalPath] = useState(project.default_spec_refs.generation_goal ?? "");
  const [goalName, setGoalName] = useState("web-generation");
  const [categories, setCategories] = useState(["normal", "boundary", "fault", "security"]);
  const [risks, setRisks] = useState(["low", "medium", "high"]);
  const [capabilities, setCapabilities] = useState("");
  const [policies, setPolicies] = useState("");
  const [mutations, setMutations] = useState("");
  const [maxCases, setMaxCases] = useState(8);
  const [instructions, setInstructions] = useState("");
  const [evidenceMode, setEvidenceMode] = useState<"MANIFEST_ONLY" | "MANIFEST_AND_CODE">("MANIFEST_ONLY");
  const [evidenceHashes, setEvidenceHashes] = useState<string[]>([]);
  const [maxOutputTokens, setMaxOutputTokens] = useState(4096);
  const [maxTotalTokens, setMaxTotalTokens] = useState("");
  const [maxCostUsd, setMaxCostUsd] = useState("");
  const [reviewer, setReviewer] = useState("local-user");
  const [reviewComment, setReviewComment] = useState("");
  const [batchCases, setBatchCases] = useState<string[]>([]);
  const [editingCase, setEditingCase] = useState<string | null>(null);
  const [editJson, setEditJson] = useState("");
  const [diffSummary, setDiffSummary] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await api.listGenerationSessions(workflow.workflow_id);
      setSessions(next);
      const active = sessionId ?? next[0]?.session_id ?? null;
      if (active) {
        setSessionId(active);
        setDetail(await api.getGenerationSession(active));
      }
    } catch (reason) {
      setError((reason as Error).message);
    }
  }, [sessionId, workflow.workflow_id]);

  useEffect(() => {
    setSessions([]); setSessionId(null); setDetail(null); setError(null);
  }, [workflow.workflow_id]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const createPlan = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError(null);
    try {
      const goal = goalMode === "new" ? {
        apiVersion: "agentqa.dev/v1alpha1",
        kind: "GenerationGoal",
        metadata: { name: goalName },
        categories,
        riskLevels: risks,
        targetCapabilities: splitValues(capabilities),
        targetPolicies: splitValues(policies),
        allowedMutations: splitValues(mutations),
        maxCases,
        instructions: instructions.trim() || undefined,
      } : undefined;
      const result = await api.createGenerationSession(workflow.workflow_id, {
        idempotencyKey: `web-plan-${Date.now()}`,
        goal,
        goalPath: goalMode === "saved" ? goalPath : undefined,
        manifestPath: project.default_spec_refs.manifest,
        mutationsPath: project.default_spec_refs.mutations,
        evidenceMode,
        codeEvidenceHashes: evidenceMode === "MANIFEST_AND_CODE" ? evidenceHashes : [],
        maxOutputTokens,
        maxTotalTokens: maxTotalTokens ? Number(maxTotalTokens) : undefined,
        maxCostUsd: maxCostUsd ? Number(maxCostUsd) : undefined,
      });
      setSessionId(result.session.session_id);
      setDetail({ session: result.session, plan: null, candidates: [] });
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const generateCandidates = async () => {
    if (!detail?.session.plan_checksum) return;
    setBusy(true); setError(null);
    try {
      await api.generateCandidates(
        detail.session.session_id,
        detail.session.plan_checksum,
        `web-tests-${Date.now()}`,
      );
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const review = async (candidate: GeneratedCandidate, decision: string) => {
    if (!detail) return;
    setBusy(true); setError(null);
    try {
      await api.reviewCandidate(detail.session.session_id, candidate.case_id, {
        subjectHash: candidate.current_spec_hash,
        decision,
        reviewer,
        comment: reviewComment,
      });
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const batchApprove = async () => {
    if (!detail || batchCases.length === 0) return;
    setBusy(true); setError(null);
    try {
      await api.batchApproveCandidates(detail.session.session_id, batchCases, reviewer, reviewComment);
      setBatchCases([]); await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const saveRevision = async (candidate: GeneratedCandidate) => {
    if (!detail) return;
    setBusy(true); setError(null);
    try {
      const spec = JSON.parse(editJson) as Record<string, unknown>;
      await api.reviseCandidate(detail.session.session_id, candidate.case_id, {
        spec, editor: reviewer, diffSummary,
      });
      setEditingCase(null); setDiffSummary(""); await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const lowRisk = useMemo(
    () => detail?.candidates.filter((item) => item.risk === "low" || item.risk === "medium") ?? [],
    [detail],
  );

  return (
    <div className="generation-workbench">
      <div className="panel__header">
        <div><p className="eyebrow">T19 · AI test generation</p><h2>Generation & review studio</h2><small>Plan first, confirm the evidence boundary, then review immutable candidates.</small></div>
        <div className="session-tabs">{sessions.map((item) => <button key={item.session_id} className={item.session_id === sessionId ? "is-selected" : ""} onClick={() => setSessionId(item.session_id)}><span>{item.status}</span><code>{item.session_id.slice(0, 8)}</code></button>)}</div>
      </div>
      {error && <div className="notice notice--bad">{error}</div>}

      <form className="generation-goal" onSubmit={(event) => void createPlan(event)}>
        <section>
          <h3>1 · Goal editor</h3>
          <div className="segmented"><button type="button" className={goalMode === "saved" ? "active" : ""} onClick={() => setGoalMode("saved")}>Saved goal</button><button type="button" className={goalMode === "new" ? "active" : ""} onClick={() => setGoalMode("new")}>New goal</button></div>
          {goalMode === "saved" ? <label>GenerationGoal path<input className="mono" value={goalPath} onChange={(event) => setGoalPath(event.target.value)} required /></label> : <div className="form-stack"><label>Goal name<input value={goalName} onChange={(event) => setGoalName(event.target.value)} required /></label><fieldset><legend>Scenario categories</legend><div className="chip-checks">{CATEGORIES.map((item) => <label key={item}><input type="checkbox" checked={categories.includes(item)} onChange={() => setCategories(toggle(categories, item))} />{item}</label>)}</div></fieldset><fieldset><legend>Risk envelope</legend><div className="chip-checks">{RISKS.map((item) => <label key={item}><input type="checkbox" checked={risks.includes(item)} onChange={() => setRisks(toggle(risks, item))} />{item}</label>)}</div></fieldset><label>Capabilities<input value={capabilities} onChange={(event) => setCapabilities(event.target.value)} placeholder="Optional, comma-separated" /></label><label>Policies<input value={policies} onChange={(event) => setPolicies(event.target.value)} placeholder="Optional, comma-separated" /></label><label>Allowed mutations<input value={mutations} onChange={(event) => setMutations(event.target.value)} placeholder="Optional, comma-separated" /></label><label>Maximum candidates<input type="number" min="1" max="50" value={maxCases} onChange={(event) => setMaxCases(Number(event.target.value))} /></label><label>Instructions<textarea value={instructions} onChange={(event) => setInstructions(event.target.value)} placeholder="Optional generation focus" /></label></div>}
        </section>
        <section>
          <h3>2 · Evidence picker</h3>
          <div className="segmented"><button type="button" className={evidenceMode === "MANIFEST_ONLY" ? "active" : ""} onClick={() => setEvidenceMode("MANIFEST_ONLY")}>Manifest only</button><button type="button" className={evidenceMode === "MANIFEST_AND_CODE" ? "active" : ""} onClick={() => setEvidenceMode("MANIFEST_AND_CODE")}>Manifest + code</button></div>
          <p className="muted">The model never browses the filesystem. Only checked, previously reviewed excerpts are sent.</p>
          {evidenceMode === "MANIFEST_AND_CODE" && <div className="evidence-options">{codeEvidence.length === 0 ? <p className="notice notice--warn">Add CodeEvidence in the project panel first.</p> : codeEvidence.map((item) => <label key={item.contentHash}><input type="checkbox" checked={evidenceHashes.includes(item.contentHash)} onChange={() => setEvidenceHashes(toggle(evidenceHashes, item.contentHash))} /><span><code>{item.path}:{item.startLine}-{item.endLine}</code><small>{item.reason}</small></span></label>)}</div>}
        </section>
        <section>
          <h3>3 · Token & cost controls</h3>
          <div className="form-stack"><label>Max output tokens / call<input type="number" min="64" max="16384" value={maxOutputTokens} onChange={(event) => setMaxOutputTokens(Number(event.target.value))} /></label><label>Total token budget<input type="number" min="1" value={maxTotalTokens} onChange={(event) => setMaxTotalTokens(event.target.value)} placeholder="Optional hard limit" /></label><label>Cost budget (USD)<input type="number" min="0" step="0.000001" value={maxCostUsd} onChange={(event) => setMaxCostUsd(event.target.value)} placeholder="Requires provider cost telemetry" /></label><p className="muted">DeepSeek currently reports tokens but not price. A configured cost limit therefore fails closed instead of estimating.</p><button className="button button--accent" disabled={busy || workflow.status !== "READY" || categories.length === 0 || risks.length === 0 || (evidenceMode === "MANIFEST_AND_CODE" && evidenceHashes.length === 0)}>Generate plan</button>{workflow.status !== "READY" && <small className="muted">The workflow must be READY before a new AI step.</small>}</div>
        </section>
      </form>

      {detail && <>
        <div className="generation-status"><span className={`status status--${detail.session.status === "FAILED_SAFE" ? "bad" : detail.session.status.endsWith("READY") ? "good" : "warn"}`}>{detail.session.status}</span><span>Evidence <strong>{detail.session.evidence_mode.replaceAll("_", " ")}</strong></span><span>Tokens <strong>{detail.session.consumed_tokens}{detail.session.max_total_tokens ? ` / ${detail.session.max_total_tokens}` : ""}</strong></span><span>Cost <strong>{detail.session.consumed_cost_usd === null ? "unobservable" : `$${detail.session.consumed_cost_usd.toFixed(6)}`}</strong></span><span>Plan <code>{shortHash(detail.session.plan_checksum)}</code></span><span>Report <code>{shortHash(detail.session.report_checksum)}</code></span></div>
        {detail.session.llm_calls.length > 0 && <div className="call-ledger">{detail.session.llm_calls.map((call, index) => <article key={`${call.evidence_hash}-${index}`}><span>AI call {index + 1}</span><strong>{call.provider} / {call.model}</strong><code>{call.status}</code><small>{call.token_usage.total_tokens ?? 0} tokens · {call.latency_ms === null ? "latency unavailable" : `${call.latency_ms.toFixed(1)} ms`} · evidence {shortHash(call.evidence_hash)}</small></article>)}</div>}
        {detail.session.error && <div className="notice notice--bad"><strong>{detail.session.error.code}</strong> {detail.session.error.message}</div>}
        {detail.plan && <section className="plan-review"><div className="review-heading"><div><p className="eyebrow">Phase one</p><h3>Plan matrix</h3><p>{detail.plan.items.length} coverage cells · {detail.plan.questions.length} open questions</p></div>{detail.session.status === "PLAN_READY" && <button className="button button--accent" disabled={busy} onClick={() => void generateCandidates()}>Confirm plan & generate candidates</button>}</div>{detail.plan.questions.length > 0 && <div className="notice notice--warn"><strong>Questions requiring judgement</strong><ul>{detail.plan.questions.map((item) => <li key={item}>{item}</li>)}</ul></div>}<div className="plan-matrix">{detail.plan.items.map((item) => <article key={item.id}><header><span className={`risk risk--${item.risk}`}>{item.risk}</span><span>{item.category}</span>{item.requires_human && <span>HITL</span>}</header><h4>{item.title}</h4><p>{item.rationale}</p><dl><dt>Capabilities</dt><dd>{item.capabilities.join(", ") || "—"}</dd><dt>Policies</dt><dd>{item.policies.join(", ") || "—"}</dd><dt>Tools</dt><dd>{item.tools.join(", ") || "—"}</dd><dt>Sources</dt><dd>{item.source_refs.join(", ")}</dd></dl></article>)}</div></section>}

        {detail.candidates.length > 0 && <section className="candidate-review"><div className="review-heading"><div><p className="eyebrow">Phase two</p><h3>Candidate review</h3><p>Original AI artifacts remain immutable. Every edit creates a checksum-linked revision.</p></div><div className="review-controls"><label>Reviewer<input value={reviewer} onChange={(event) => setReviewer(event.target.value)} /></label><label>Review note<input value={reviewComment} onChange={(event) => setReviewComment(event.target.value)} /></label><button className="button button--small" disabled={busy || batchCases.length === 0} onClick={() => void batchApprove()}>Approve selected low/medium ({batchCases.length})</button></div></div><div className="candidate-list">{detail.candidates.map((candidate) => <article className="candidate-card" key={candidate.case_id}><header><label className="candidate-select">{(candidate.risk === "low" || candidate.risk === "medium") && <input type="checkbox" checked={batchCases.includes(candidate.case_id)} onChange={() => setBatchCases(toggle(batchCases, candidate.case_id))} />}</label><div><h4>{candidate.case_id}</h4><p>{candidate.rationale}</p></div><span className={`risk risk--${candidate.risk}`}>{candidate.risk}</span><span className={`status ${candidate.campaign_eligible ? "status--good" : "status--warn"}`}>{candidate.campaign_eligible ? "CAMPAIGN READY" : "BLOCKED"}</span></header><div className="candidate-columns"><section><h5>Assertions & provenance</h5><ul>{candidate.assertion_basis.map((item) => <li key={item}>{item}</li>)}</ul><p><strong>Coverage</strong> {candidate.coverage_item_ids.join(", ")}</p><p><strong>Sources</strong> {candidate.source_refs.join(", ")}</p></section><section><h5>Diagnostics</h5>{candidate.diagnostics.map((item) => <p key={item.code}><span className={`status status--${item.status === "PASS" ? "good" : item.status === "BLOCK" ? "bad" : "warn"}`}>{item.status}</span> <strong>{item.code}</strong> {item.message}</p>)}{candidate.similarity_hints.map((item) => <p key={item.source}>Similarity {Math.round(item.score * 100)}% with <code>{item.source}</code></p>)}</section><section><h5>Identity</h5><p>Current <code>{shortHash(candidate.current_spec_hash)}</code></p><p>Original <code>{shortHash(candidate.original_spec_hash)}</code></p>{candidate.latest_revision && <p>Revision {candidate.latest_revision.revision} by {candidate.latest_revision.editor}</p>}{candidate.decision && <p><strong>{candidate.decision.decision}</strong> by {candidate.decision.reviewer}</p>}</section></div><details><summary>Scenario, expectations, evaluators and budgets</summary><pre>{JSON.stringify(candidate.spec, null, 2)}</pre></details>{editingCase === candidate.case_id ? <div className="revision-editor"><label>Edited AgentTest JSON<textarea className="mono" value={editJson} onChange={(event) => setEditJson(event.target.value)} /></label><label>Diff summary<input value={diffSummary} onChange={(event) => setDiffSummary(event.target.value)} required /></label><div className="button-row"><button className="button button--small button--accent" disabled={busy || !diffSummary.trim()} onClick={() => void saveRevision(candidate)}>Save immutable revision</button><button className="button button--small" onClick={() => setEditingCase(null)}>Cancel</button></div></div> : <div className="candidate-actions"><button className="button button--small" disabled={busy} onClick={() => void review(candidate, "APPROVED")}>Approve</button><button className="button button--small" disabled={busy} onClick={() => void review(candidate, "REJECTED")}>Reject</button><button className="button button--small" disabled={busy} onClick={() => void review(candidate, "REQUEST_EDIT")}>Request edit</button><button className="button button--small" disabled={busy} onClick={() => { setEditingCase(candidate.case_id); setEditJson(JSON.stringify(candidate.spec, null, 2)); }}>Edit as revision</button>{(candidate.risk === "high" || candidate.risk === "critical") && <small>High-risk approval is intentionally individual.</small>}</div>}</article>)}</div></section>}
      </>}
    </div>
  );
}
