# Result storage, local API, and evidence console

T9 adds a local persistence and inspection layer without changing the immutable
`RunResult` produced by the runner. `StoredRun` wraps that evidence with a
project identifier and an optional artifact directory, so existing CLI
artifacts remain the source of truth and older consumers do not gain storage
concerns.

## ResultStore

`ResultStore` is an asynchronous protocol with operations for runs, individual
test results, traces, findings, and review state. Two implementations are
provided:

- `FileResultStore` writes canonical JSON documents, adjacent SHA-256 files,
  and an atomically replaced index. It is useful for portable local evidence.
- `SQLiteResultStore` uses the Python standard library, foreign keys, WAL mode,
  a busy timeout, and explicit transactions. It is the default API backend.

Both stores redact recognized secrets before persistence, reject conflicting
reuse of a run ID, and keep large traces and replay cassettes in their original
artifact directory. The database stores only the artifact root and relative
trace reference. Trace reads enforce containment, verify the adjacent checksum,
and validate the `TraceRecord` schema.

SQLite creates these normalized tables:

| Table | Purpose |
| --- | --- |
| `projects` | Local project namespace |
| `runs` | Run lifecycle, summary, and immutable stored aggregate |
| `test_results` | Searchable trial rows and serialized result records |
| `traces` | Trace status and artifact path references |
| `findings` | Stable evaluator-violation records |
| `review_status` | Mutable human judgement separated from evidence |

The schema version is recorded in `schema_metadata`. An unknown version fails
closed; T9 does not attempt an implicit migration.

## Local API

Install the optional server dependencies and start the application:

```bash
python -m pip install -e ".[api]"
agentqa serve --workspace . --data .agentqa --artifacts .agentqa/artifacts
```

The server binds to loopback by default. A non-loopback `--host` is rejected
unless `--allow-remote` is explicit because T9 intentionally has no
authentication or multi-tenant isolation. Run and gate specification paths
must be relative to `--workspace`; absolute paths, `..`, and symlink escapes
are rejected.

The versioned application exposes OpenAPI at `/docs` and these primary routes:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Storage and workspace health |
| `GET` | `/api/runs?project_id=default` | Final runs plus active local jobs |
| `POST` | `/api/runs` | Submit one bounded background run |
| `GET` | `/api/runs/{run_id}` | Poll a job or read its stored aggregate |
| `GET` | `/api/runs/{run_id}/results` | Read test-trial evidence |
| `GET` | `/api/ai/evidence` | Read checksummed generation and AnalysisReport summaries |
| `GET` | `/api/runs/{run_id}/findings` | Read stable findings |
| `PUT` | `/api/findings/{finding_id}/review` | Save human review state |
| `GET` | `/api/traces/{trace_id}` | Read checked artifact or live trace data |
| `POST` | `/api/comparisons` | Compare two stored artifact runs |
| `POST` | `/api/gates/evaluate` | Evaluate a workspace gate against stored runs |
| `POST` | `/v1/traces` | Receive size-limited OTLP/HTTP protobuf traces |

T17 adds the persistent workbench control plane alongside the existing run
ledger:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` / `POST` | `/api/projects` | List or create local workbench projects |
| `GET` / `POST` | `/api/workflows` | List or idempotently create workflows |
| `GET` | `/api/workflows/{workflow_id}` | Read a workflow and its step attempts |
| `POST` | `/api/workflows/{workflow_id}/resume` | Resume from a persisted safe checkpoint |
| `POST` | `/api/workflows/{workflow_id}/cancel` | Cancel without deleting prior evidence |
| `GET` | `/api/workflows/{workflow_id}/events?after=N` | Read revision-ordered audit events |

Workflow requests use an idempotency key. Reusing the same key and identical
input returns the original workflow; reusing it with changed input returns a
conflict. State and step input are written transactionally, recognized secrets
are redacted before persistence, and every state change increments the event
revision. On API startup, an unfinished `RUNNING` workflow is changed to
`INTERRUPTED` rather than guessed complete. It can then be explicitly resumed
to `READY`. AI and campaign steps are independently bounded to one concurrent
operation by default. T17 provides this orchestration boundary; later workbench
tasks connect artifact import, generation, review, campaign, and analysis
actions to it.

T18 adds project readiness, a verified artifact index, and explicit code
evidence selection. These routes never accept absolute paths and never move or
rewrite imported artifact files:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/projects/{project_id}` | Read project configuration and the last observed source snapshot |
| `POST` | `/api/projects/{project_id}/preflight` | Validate specs, paths, entrypoints, dependencies, provider state, tool risk, side effects, and local source identity |
| `POST` | `/api/projects/{project_id}/artifacts/import` | Explicitly discover checksummed artifacts in allowlisted workspace roots |
| `GET` | `/api/projects/{project_id}/artifacts` | List catalog entries independently of their original directory layout |
| `GET` | `/api/projects/{project_id}/workspace?path=.` | Browse only safe text files under the configured project directory |
| `POST` | `/api/projects/{project_id}/code-evidence/preview` | Preview a bounded, redacted file and line selection without granting AI access |
| `POST` | `/api/projects/{project_id}/code-evidence` | Explicitly add the previewed selection to the project's immutable evidence set |
| `GET` | `/api/projects/{project_id}/code-evidence` | List selected code evidence and content hashes |
| `GET` | `/api/projects/{project_id}/evidence-preview` | Show the exact current AI evidence boundary and budgets |

The discovery roots are `.agentqa/artifacts`, the configured artifact root,
`.agentqa/mcp`, `benchmark/results`, and `benchmark/reference-results`.
Discovery requires an adjacent `.sha256` file, enforces a 16 MiB per-file
limit, checks the recognized artifact schema, and records only an index row.
Missing files become `MISSING`; changed files or sidecars become `INVALID`.
Complete imported run directories are also reconstructed into `ResultStore`,
which makes older CLI and T16 runs visible on the Runs page. AI Evidence reads
catalog entries rather than depending on a fixed directory depth. Catalog reads
are bounded to 200 rows by default (maximum 500) and return `total`, `limit`,
and `offset` alongside the current page.

Code evidence uses a separate two-step boundary. Preview rejects `.env*`, Git
metadata, credentials, keys, binary or oversized files, generated/dependency
directories, traversal, and symlinks. It returns only a redacted excerpt and
its content hash. A maximum of 10 explicitly selected excerpts and 160 KiB of
redacted content is retained per project. The evidence preview lists included
Manifest sections and excerpts, excluded data classes, provider/model, token
limits, timeout, and a deterministic evidence hash before a later AI action is
allowed to use it.

T19 connects the existing two-phase generator to persistent Workflow steps and
the browser review surface:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` / `POST` | `/api/workflows/{workflow_id}/generation-sessions` | List sessions or queue an asynchronous Plan step |
| `GET` | `/api/generation-sessions/{session_id}` | Read session budget, Plan, candidates, diagnostics, reviews, and revisions |
| `POST` | `/api/generation-sessions/{session_id}/candidates` | Confirm the checksummed Plan and queue candidate generation |
| `POST` | `/api/generation-sessions/{session_id}/candidates/{case_id}/review` | Record a checksum-bound approve, reject, or request-edit decision |
| `POST` | `/api/generation-sessions/{session_id}/candidates/batch-approve` | Approve only low/medium-risk current revisions in a batch |
| `POST` | `/api/generation-sessions/{session_id}/candidates/{case_id}/revisions` | Create a validated immutable edit with parent and new hashes |

The first two LLM phases are deliberately separate. A session persists only
bounded goal/evidence identities, safe LLM call accounting, token/cost state,
and references to checksummed artifacts; raw prompts and provider output are
not workflow metadata. Manifest-only mode clears code context. Code-assisted
mode accepts only hashes already recorded by the T18 CodeEvidence picker.
Generation fails closed for provider errors, invalid structured output, token
overruns, and an unobservable configured cost ceiling.

`GenerationPlan`, `GenerationReport`, and generated YAML files are write-once.
Human edits are separate revision rows and never rewrite those originals.
Approval is also a separate record bound to the current spec hash, so any edit
automatically invalidates an earlier decision. High/critical candidates remain
campaign-ineligible until individually approved; the server rejects attempts
to include them in low/medium batch approval. Similarity is advisory only and
never silently deletes a candidate.

`POST /api/runs` returns `202 Accepted` and a stable run ID immediately. The
client polls the run listing or detail route until the state is finalized. Jobs
are in-process and intentionally do not survive a server restart.

The run request accepts workspace-relative `testsPath`, `targetPath`, optional
`mutationsPath`, and optional `manifestPath`. Manifest paths must name a file
inside the configured workspace. CLI, API, and Web launches all enter the same
application service and therefore produce the same `resolved-manifest.json`
contract and drift status.

When `.env.local` contains valid DeepSeek structured-LLM settings,
`/api/health` reports
the configured provider and model without exposing credentials. Set
`judgeEnabled: true` on `POST /api/runs` to run `llm_judge` evaluators. A request
that enables Judge while the server has no provider is rejected before a job is
created.

## React evidence console

The Vite/React TypeScript application lives in `apps/web`. It supports the run
ledger, run summaries, test results, trace trees, explicit not-observable
states, findings, mutation evidence, comparisons, and quality-gate decisions.
The launch form accepts an optional Manifest path and exposes a DeepSeek opt-in
when the server is configured. Run details show each AI verdict, score, and
rationale.
The **Workbench** view creates projects and durable workflow containers, polls
their current state, shows the ordered audit timeline, and exposes only valid
resume and cancel actions. It also provides T18 project setup, Preflight,
artifact discovery/catalog inspection, safe workspace browsing, redacted code
preview with an explicit add action, and evidence-boundary inspection. The
T19 Generation & Review Studio adds a Goal Editor, reviewed-evidence picker,
Plan Matrix, explicit Plan confirmation, Token/Cost summary, candidate
diagnostics, risk approvals, and immutable JSON revision editor. The existing
**Runs** view and CLI remain compatible;
workflow metadata shares the SQLite file but uses separate versioned tables.
The **AI evidence** view additionally discovers strict, checksummed generation
reports and AnalysisReports from the MCP and T16 artifact layouts. It shows
Manifest/plan/context hashes, model calls and token usage, generated-case risk
and approval status, and ranked evidence-citing root causes. This endpoint is
read-only and ignores missing, malformed, oversized, or checksum-invalid files.
It polls every two seconds rather than requiring a WebSocket and remains usable
at the 1280 by 720 acceptance viewport as well as narrow screens.

For development, run the API and Vite separately:

```bash
agentqa serve --static /path/that/does/not/exist
cd apps/web
npm ci
npm run dev
```

Vite proxies `/api` and `/v1` to `127.0.0.1:8000`. For a single production-like
local process, run `npm run build` and then `agentqa serve`; the CLI serves
`apps/web/dist` when it exists. No remote fonts, analytics, editor, WebSocket,
or account service is required.

Run Web verification with:

```bash
npm test
npm run build
```
