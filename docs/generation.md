# AI-assisted test generation

AgentQAKit generates candidate tests in two explicit phases. It does not scan a
repository or execute generated code. The caller supplies an `AgentManifest`, an
optional `GenerationGoal`, and optionally selected `CodeEvidence` excerpts.

## Why generation has two phases

The plan phase decides what should be tested before producing executable test
specifications. Each plan item binds a scenario category and risk level to
declared capabilities, policies, tools, optional mutations, and evidence
references. Missing information is recorded as a question instead of being
invented.

The test phase converts the accepted plan into strict `AgentTest` drafts. Every
draft must name its covered plan items, evidence sources, rationale, and
assertion basis. AgentQAKit then validates the draft independently of the model.

```text
AgentManifest + GenerationGoal + optional CodeEvidence
                         |
                         v
                  GenerationPlan
                         |
                         v
              candidate AgentTest specs
                         |
                         v
 schema -> Pydantic -> manifest references -> evaluator/mutation checks
        -> risk policy -> semantic deduplication -> checksummed artifacts
```

## Generate a plan

With an explicit goal:

```bash
agentqa generate plan \
  --manifest examples/specs/agent-manifest.yaml \
  --goal examples/specs/generation-goal.yaml \
  --output artifacts/generation/plan.json
```

Omit `--goal` to use a bounded default goal. DeepSeek configuration is loaded
from `.env.local` by default and can be changed with `--llm-env-file`. Keep
`AGENTQA_LLM_MAX_TOKENS=8192` for generation; the more compact Judge workflow
still applies its lower per-call cap. Keep
`AGENTQA_LLM_REASONING_EFFORT=none` for schema-heavy JSON generation unless a
larger reasoning-inclusive output budget is deliberately configured. The plan
and its adjacent `.sha256` file are immutable inputs to the next phase.

Plan categories are `normal`, `boundary`, `fault`, `security`, `efficiency`, and
`hitl`. Risk levels are `low`, `medium`, `high`, and `critical`. A goal can also
limit generation to named capabilities or policies and allow named Mutation
specifications.

For policy lists that do not carry explicit IDs, AgentQAKit exposes stable plan
identifiers in document order: `forbidden-behavior-1` and
`permission-requirement-1`. Invariants retain their declared `id`.

## Generate candidate tests

```bash
agentqa generate tests \
  --manifest examples/specs/agent-manifest.yaml \
  --plan artifacts/generation/plan.json \
  --output tests/generated

agentqa test validate tests/generated
```

If the plan allows mutations, pass the exact available definitions:

```bash
agentqa generate tests \
  --manifest examples/specs/agent-manifest.yaml \
  --plan artifacts/generation/plan.json \
  --mutations examples/specs/mutations.yaml \
  --output tests/generated
```

The output directory must be absent or empty and must remain inside
`--workspace` (the current directory by default). It contains one YAML file and
checksum per generated test, plus `generation-report.json` and its checksum.
The report maps every declared capability and policy to generated case IDs and
lists uncovered subjects as gaps.

Generated tests omit `targetRef`; select a target at run time:

```bash
agentqa test run tests/generated \
  --target examples/specs/python-target.yaml \
  --manifest examples/specs/agent-manifest.yaml
```

## Review and approval

AgentQAKit assigns controlled labels after generation. Low- and medium-risk
tests receive `review-status: READY`. High- and critical-risk tests receive
`review-status: REVIEW_REQUIRED`, and the run service refuses to execute any
such test.

Review the generated YAML, its rationale and assertion basis in
`generation-report.json`, select an appropriate target, and make any required
edits. Approval is explicit: change the generated test label to
`review-status: APPROVED`, then validate the edited YAML before running it. The
original checksum will no longer match after a human edit; this is intentional
evidence that the generated draft was changed.

## CodeEvidence

`CodeEvidence` lets a coding agent or developer attach only a relevant excerpt.
It contains a safe workspace-relative POSIX path, inclusive line range, reason,
bounded content, and `sha256:` content hash. AgentQAKit verifies the hash before
calling the model. Raw excerpt content is used as untrusted prompt evidence but
is not copied into `GenerationPlan` or `GenerationReport`; those artifacts keep
only the content hash.

This boundary avoids giving AgentQAKit ambient filesystem access while still
letting an external coding agent contribute code-aware context.

## Safety and validation rules

The generation service rejects a result when it:

- references an undeclared capability, policy, tool, source, or plan item;
- uses a Mutation not allowed by the goal and supplied to the test phase;
- uses an unsupported evaluator or relies on `llm_judge` alone;
- includes `targetRef`, template expressions, or executable template content;
- duplicates another generated scenario/expectation/mutation combination;
- requests human interaction that the Manifest does not allow; or
- exceeds the goal, prompt, response, evidence, or artifact size bounds.

Invalid structured model output is retried once and then fails closed. Persisted
AI evidence contains provider/model identity, parameters, hashes, request IDs,
latency, attempts, and token usage—not API keys or raw prompts.

The reusable application boundary is `GenerationService`. CLI commands are thin
adapters over `generate_plan()` and `generate_tests()`, so later API and MCP
tools can use the same validation and policy behavior.
