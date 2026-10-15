import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type { Campaign, CampaignInputs, WorkflowRun } from "../types";

function statusClass(status: string) {
  if (["COMPLETED", "PASS"].includes(status)) return "status--good";
  if (["FAILED", "CANCELLED"].includes(status)) return "status--bad";
  if (["RUNNING", "INCOMPLETE", "INTERRUPTED", "AWAITING_CONFIRMATION"].includes(status)) return "status--warn";
  return "status--neutral";
}

function short(value: string) {
  return value.length > 20 ? `${value.slice(0, 10)}…${value.slice(-8)}` : value;
}

function elapsed(start: string | null, end: string | null) {
  if (!start) return "—";
  const milliseconds = Math.max(0, new Date(end ?? Date.now()).getTime() - new Date(start).getTime());
  return `${(milliseconds / 1000).toFixed(1)}s`;
}

export function CampaignWorkbench({ workflow }: { workflow: WorkflowRun }) {
  const [inputs, setInputs] = useState<CampaignInputs | null>(null);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [selectedGolden, setSelectedGolden] = useState<string[]>([]);
  const [selectedGenerated, setSelectedGenerated] = useState<string[]>([]);
  const [selectedTargets, setSelectedTargets] = useState<string[]>([]);
  const [mutations, setMutations] = useState(false);
  const [replayMode, setReplayMode] = useState<"off" | "record" | "replay" | "hybrid">("off");
  const [cassette, setCassette] = useState("");
  const [judge, setJudge] = useState(false);
  const [trials, setTrials] = useState(1);
  const [concurrency, setConcurrency] = useState(1);
  const [seed, setSeed] = useState(0);
  const [timeout, setTimeoutValue] = useState(120);
  const [maxTokens, setMaxTokens] = useState("");
  const [maxCost, setMaxCost] = useState("");
  const [maxDuration, setMaxDuration] = useState("");
  const [allowSideEffects, setAllowSideEffects] = useState(false);
  const [selectedCampaignId, setSelectedCampaignId] = useState<string | null>(null);
  const [challenge, setChallenge] = useState<{ id: string; phrase: string; configHash: string } | null>(null);
  const [confirmation, setConfirmation] = useState("");
  const [reviewer, setReviewer] = useState("web-user");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [nextInputs, nextCampaigns] = await Promise.all([
        api.campaignInputs(workflow.workflow_id),
        api.listCampaigns(workflow.workflow_id),
      ]);
      setInputs(Array.isArray(nextInputs.golden) && Array.isArray(nextInputs.targets) ? nextInputs : null);
      setCampaigns(nextCampaigns);
      setSelectedCampaignId((current) => current ?? nextCampaigns[0]?.campaign_id ?? null);
      setError(null);
    } catch (reason) {
      setError((reason as Error).message);
    }
  }, [workflow.workflow_id]);

  useEffect(() => {
    setInputs(null); setCampaigns([]); setSelectedCampaignId(null); setChallenge(null);
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!inputs) return;
    setSelectedGolden((current) => current.length ? current : inputs.golden.map((item) => item.caseId));
    setSelectedTargets((current) => current.length ? current : inputs.targets.filter((item) => item.role === "fixed" || item.role === "target").map((item) => `${item.role}:${item.path}`));
  }, [inputs]);

  const selected = useMemo(
    () => campaigns.find((item) => item.campaign_id === selectedCampaignId) ?? null,
    [campaigns, selectedCampaignId],
  );

  const toggle = (value: string, values: string[], update: (next: string[]) => void) => {
    update(values.includes(value) ? values.filter((item) => item !== value) : [...values, value]);
  };

  const create = async () => {
    if (!inputs) return;
    setBusy(true); setError(null); setChallenge(null);
    try {
      const campaign = await api.createCampaign(workflow.workflow_id, {
        idempotencyKey: `campaign-${Date.now()}`,
        goldenTestsPath: inputs.goldenTestsPath,
        goldenCaseIds: selectedGolden,
        generatedTests: inputs.generated
          .filter((item) => selectedGenerated.includes(`${item.sessionId}:${item.caseId}`))
          .map((item) => ({ sessionId: item.sessionId, caseId: item.caseId, specHash: item.specHash })),
        targets: inputs.targets.filter((item) => selectedTargets.includes(`${item.role}:${item.path}`)),
        manifestPath: inputs.manifestPath,
        mutationsPath: mutations ? inputs.mutationsPath : undefined,
        replayMode,
        replayCassette: replayMode === "replay" || replayMode === "hybrid" ? cassette : undefined,
        judgeEnabled: judge,
        trials,
        randomSeed: seed,
        concurrency,
        itemTimeoutSeconds: timeout,
        evaluatorTimeoutSeconds: Math.min(timeout, 120),
        maxTotalTokens: maxTokens ? Number(maxTokens) : undefined,
        maxTotalCostUsd: maxCost ? Number(maxCost) : undefined,
        maxDurationSeconds: maxDuration ? Number(maxDuration) : undefined,
        allowSideEffects,
      });
      setCampaigns((current) => [campaign, ...current.filter((item) => item.campaign_id !== campaign.campaign_id)]);
      setSelectedCampaignId(campaign.campaign_id);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const requestChallenge = async () => {
    if (!selected) return;
    setBusy(true); setError(null);
    try {
      const value = await api.createCampaignChallenge(selected.campaign_id);
      setChallenge({ id: value.challenge.challenge_id, phrase: value.phrase, configHash: value.challenge.config_hash });
      setConfirmation("");
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const confirm = async () => {
    if (!selected || !challenge) return;
    setBusy(true); setError(null);
    try {
      const campaign = await api.confirmCampaign(selected.campaign_id, {
        challengeId: challenge.id, phrase: confirmation, configHash: challenge.configHash,
        reviewer, comment: "Confirmed in Campaign Builder after reviewing risk summary.",
      });
      setCampaigns((current) => current.map((item) => item.campaign_id === campaign.campaign_id ? campaign : item));
      setChallenge(null); setConfirmation("");
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  const start = async () => {
    if (!selected) return;
    setBusy(true); setError(null);
    try {
      const value = await api.startCampaign(selected.campaign_id, selected.config_hash, `start-${selected.campaign_id}-${Date.now()}`);
      setCampaigns((current) => current.map((item) => item.campaign_id === value.campaign.campaign_id ? value.campaign : item));
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };

  return <section className="campaign-workbench">
    <div className="panel__header"><div><p className="eyebrow">T20 execution workspace</p><h2>Campaign builder</h2><small>Freeze reviewed tests, targets, budgets, mutation and replay policy before execution.</small></div><div className="session-tabs">{campaigns.map((item) => <button key={item.campaign_id} className={selectedCampaignId === item.campaign_id ? "is-selected" : ""} onClick={() => { setSelectedCampaignId(item.campaign_id); setChallenge(null); }}><span>{item.status}</span><code>{short(item.campaign_id)}</code></button>)}</div></div>
    {error && <div className="notice notice--bad">{error}</div>}
    {!inputs ? <p className="padded muted">Loading selectable test revisions and targets…</p> : <>
      <div className="campaign-builder-grid">
        <section><h3>1. Test revisions</h3><p className="muted">Golden tests and only campaign-eligible generated revisions can be frozen together.</p><div className="campaign-options">{inputs.golden.map((item) => <label key={item.caseId}><input type="checkbox" checked={selectedGolden.includes(item.caseId)} onChange={() => toggle(item.caseId, selectedGolden, setSelectedGolden)} /><span><strong>{item.caseId}</strong><small>golden · {item.risk} · {item.specHash.slice(7, 19)}</small></span></label>)}{inputs.generated.map((item) => { const key = `${item.sessionId}:${item.caseId}`; return <label key={key} className={!item.campaignEligible ? "is-disabled" : ""}><input type="checkbox" disabled={!item.campaignEligible} checked={selectedGenerated.includes(key)} onChange={() => toggle(key, selectedGenerated, setSelectedGenerated)} /><span><strong>{item.caseId}</strong><small>generated · {item.risk} · {item.reviewStatus}</small></span></label>; })}</div></section>
        <section><h3>2. Target matrix</h3><p className="muted">Every selected target receives an independent run ID and artifact directory.</p><div className="campaign-options">{inputs.targets.map((item) => { const key = `${item.role}:${item.path}`; return <label key={key}><input type="checkbox" checked={selectedTargets.includes(key)} onChange={() => toggle(key, selectedTargets, setSelectedTargets)} /><span><strong>{item.role}</strong><small className="mono">{item.path}</small></span></label>; })}</div><label className="check-row"><input type="checkbox" checked={mutations} disabled={!inputs.mutationsPath} onChange={(event) => setMutations(event.target.checked)} /><span>Mutation campaign<small>{inputs.mutationsPath ?? "No project MutationSpec configured"}</small></span></label></section>
        <section><h3>3. Runtime policy</h3><div className="form-stack"><label>Replay mode<select value={replayMode} onChange={(event) => setReplayMode(event.target.value as typeof replayMode)}><option value="off">off</option><option value="record">record</option><option value="replay">replay (fail closed)</option><option value="hybrid">hybrid (miss may call live)</option></select></label>{(replayMode === "replay" || replayMode === "hybrid") && <label>Cassette path<input className="mono" value={cassette} onChange={(event) => setCassette(event.target.value)} placeholder=".agentqa/artifacts/runs/.../cassettes/....jsonl" /></label>}<div className="campaign-numbers"><label>Trials<input type="number" min="1" max="100" value={trials} onChange={(event) => setTrials(Number(event.target.value))} /></label><label>Concurrency<input type="number" min="1" max="32" value={concurrency} onChange={(event) => setConcurrency(Number(event.target.value))} /></label><label>Seed<input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label><label>Item timeout<input type="number" min="1" value={timeout} onChange={(event) => setTimeoutValue(Number(event.target.value))} /></label></div><label>Campaign max duration (seconds)<input type="number" min="1" value={maxDuration} onChange={(event) => setMaxDuration(event.target.value)} placeholder="optional" /></label><div className="campaign-numbers"><label>Max tokens<input type="number" min="1" value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} /></label><label>Max cost USD<input type="number" min="0" step="0.01" value={maxCost} onChange={(event) => setMaxCost(event.target.value)} /></label></div><label className="check-row"><input type="checkbox" checked={judge} disabled={!inputs.judge.available} onChange={(event) => setJudge(event.target.checked)} /><span>DeepSeek Judge<small>{inputs.judge.available ? `${inputs.judge.provider} / ${inputs.judge.model}` : "provider unavailable"}</small></span></label><label className="check-row"><input type="checkbox" checked={allowSideEffects} onChange={(event) => setAllowSideEffects(event.target.checked)} /><span>Allow declared external side effects<small>Still requires a server-verified one-time confirmation.</small></span></label></div></section>
      </div>
      <div className="campaign-create"><span><strong>{selectedGolden.length + selectedGenerated.length}</strong> selected cases · <strong>{selectedTargets.length}</strong> targets · up to <strong>{(selectedGolden.length + selectedGenerated.length) * selectedTargets.length * trials}</strong> base trials{mutations ? " before mutation expansion" : ""}</span><button className="button button--accent" disabled={busy || selectedTargets.length === 0 || selectedGolden.length + selectedGenerated.length === 0 || ((replayMode === "replay" || replayMode === "hybrid") && !cassette)} onClick={() => void create()}>Freeze campaign config</button></div>
    </>}

    {selected && <div className="campaign-review">
      <div className="review-heading"><div><p className="eyebrow">Immutable launch summary</p><h3>{selected.risk.case_count} cases × {selected.risk.target_count} targets</h3><p><code>{selected.config_hash}</code></p></div><span className={`status ${statusClass(selected.status)}`}>{selected.status}</span></div>
      <div className="campaign-risk-grid"><span>Maximum trials<strong>{selected.risk.max_trials}</strong></span><span>Mutations<strong>{selected.risk.mutation_count}</strong></span><span>High risk<strong>{selected.risk.high_risk_cases.length}</strong></span><span>Side-effect tools<strong>{selected.risk.side_effect_tools.length}</strong></span><span>Replay<strong>{selected.config.replay_mode}</strong></span><span>Judge<strong>{selected.config.judge_enabled ? "enabled" : "off"}</strong></span></div>
      {(selected.risk.reasons.length > 0 || selected.risk.side_effect_tools.length > 0) && <div className="notice notice--warn"><strong>Confirmation scope:</strong> {[...selected.risk.reasons, ...selected.risk.side_effect_tools.map((tool) => `tool ${tool}`)].join(" · ")}</div>}
      {selected.status === "AWAITING_CONFIRMATION" && !challenge && <div className="campaign-actions"><button className="button" disabled={busy} onClick={() => void requestChallenge()}>Generate one-time confirmation</button></div>}
      {challenge && <div className="confirmation-box"><p>Review the summary, then type <code>{challenge.phrase}</code>. It expires in 10 minutes and cannot be reused.</p><div className="inline-form"><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder={challenge.phrase} /><button className="button button--accent" disabled={busy || confirmation !== challenge.phrase || !reviewer} onClick={() => void confirm()}>Confirm exact config</button></div><label>Reviewer<input value={reviewer} onChange={(event) => setReviewer(event.target.value)} /></label></div>}
      {(["DRAFT", "INTERRUPTED", "INCOMPLETE"] as string[]).includes(selected.status) && <div className="campaign-actions"><button className="button button--accent" disabled={busy} onClick={() => void start()}>{selected.status === "DRAFT" ? "Start campaign" : "Retry from safe checkpoint"}</button></div>}
      <div className="campaign-progress"><div><span>Progress</span><strong>{selected.completed_trials} / {selected.total_trials} trials</strong></div><progress max={selected.total_trials} value={selected.completed_trials} /></div>
      <div className="table-scroll"><table className="campaign-matrix"><thead><tr><th>Target</th><th>Status</th><th>Run / attempt</th><th>Trials</th><th>Case</th><th>Elapsed</th><th>Verdict</th><th>Judge</th><th>Replay H/M/live</th><th>Mutation apps · K/S/NR/I</th><th>Budget</th></tr></thead><tbody>{selected.subruns.map((subrun) => <tr key={subrun.subrun_id}><td><strong>{subrun.target.role}</strong><small>{subrun.target.name}</small></td><td><span className={`status ${statusClass(subrun.status)}`}>{subrun.status}</span></td><td><code title={subrun.run_id}>{short(subrun.run_id)}</code><small>attempt {subrun.attempt}</small></td><td>{subrun.completed_trials}/{subrun.planned_trials}</td><td>{subrun.current_case_id ?? "—"}</td><td>{elapsed(subrun.started_at, subrun.finished_at)}</td><td>{subrun.summary ? `${subrun.summary.passed} pass / ${subrun.summary.failed} fail / ${subrun.summary.incomplete} incomplete` : "not final"}</td><td>{subrun.judge_evaluations}</td><td>{subrun.replay_hits}/{subrun.replay_misses}/{subrun.live_fallbacks}</td><td>{subrun.mutation_applications} · {subrun.mutation_killed}/{subrun.mutation_survived}/{subrun.mutation_not_reached}/{subrun.mutation_invalid}</td><td>{subrun.budget ? `${subrun.budget.consumed_total_tokens} tok · $${subrun.budget.consumed_cost_usd.toFixed(4)}` : "unobserved"}</td></tr>)}</tbody></table></div>
      {selected.error && <div className="notice notice--bad"><strong>{selected.error.code}</strong> — {selected.error.message}</div>}
      <p className="campaign-footnote">Open the Runs tab with a run ID to inspect evaluator decisions, trace trees, state/tool evidence, replay details, and verified failure bundles.</p>
    </div>}
  </section>;
}
