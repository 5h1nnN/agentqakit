# T22 real Web acceptance reference

This directory contains the bounded reference output from the real Web Golden Path completed on 2026-10-16 (Asia/Shanghai).

- Provider/model: DeepSeek / `deepseek-flash`
- Project: `t22-real-20261016f`
- Workflow: `30748b2c-4a42-4bae-b578-0f2fc46b9fb8`
- Campaigns: the original 33-trial record/mutation matrix, one fail-closed replay trial,
  and an 11-trial fixed-target mutation campaign; 45 trials across 5 runs in total
- Result: the authorization defect failed all 11 trials while target and fixed roles
  passed all 11; the replay recorded 1 hit, 0 misses, and 0 live fallbacks
- Fixed-target mutation score: `0.8` (4 killed, 1 survived, 6 not reached)
- Gates: the final regression and mutation gates both passed with exit code 0
- AI evidence: one persisted failure analysis, with explicit evidence authorization and a checksummed handoff resource

An earlier mutation-gate attempt against the authorization-only fixed run is intentionally
retained in the audit report as a failed-safe result: no mutation was reached, so the score
was unobservable. The dedicated `course-ai101` mutation campaign supplied the missing
evidence and the final mutation gate passed. This documents the recovery path instead of
hiding an intermediate decision.

`workflow-report.json.sha256` authenticates the downloaded report. `acceptance.json` records the Web channel, provider, project, completion time, and report checksum. The directory passed `scripts/scan_artifact_leaks.py`; no API key, authorization header, raw prompt, raw code excerpt, or raw trace content is included.
