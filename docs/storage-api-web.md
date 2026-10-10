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
| `GET` | `/api/runs/{run_id}/findings` | Read stable findings |
| `PUT` | `/api/findings/{finding_id}/review` | Save human review state |
| `GET` | `/api/traces/{trace_id}` | Read checked artifact or live trace data |
| `POST` | `/api/comparisons` | Compare two stored artifact runs |
| `POST` | `/api/gates/evaluate` | Evaluate a workspace gate against stored runs |
| `POST` | `/v1/traces` | Receive size-limited OTLP/HTTP protobuf traces |

`POST /api/runs` returns `202 Accepted` and a stable run ID immediately. The
client polls the run listing or detail route until the state is finalized. Jobs
are in-process and intentionally do not survive a server restart.

## React evidence console

The Vite/React TypeScript application lives in `apps/web`. It supports the run
ledger, run summaries, test results, trace trees, explicit not-observable
states, findings, mutation evidence, comparisons, and quality-gate decisions.
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
