# Changelog

All notable changes to AgentQAKit are documented in this file. The project uses
[Semantic Versioning](https://semver.org/) and PEP 440 pre-release identifiers.

## [Unreleased]

## [0.1.0a0] - 2026-10-21

### Added

- Strict, versioned Agent Manifest, Test, Target, Mutation, Generation Goal, and
  Quality Gate specifications with exported JSON Schemas.
- Budget-aware concurrent runner with HTTP, isolated Python, Python SDK, and
  optional LangGraph target adapters.
- OpenTelemetry tracing, deterministic fault injection, Mutation Score,
  record/replay, verified failure bundles, run comparison, and release gates.
- Optional structured LLM judge, two-phase AI test generation, and
  evidence-grounded failure analysis.
- Local FastAPI and React workbench plus an approval-gated stdio MCP server.
- AgentQA-Bench, the Campus Ops reference Agent, CI examples, and checksummed
  reference results.
- Reproducible wheel and source-distribution workflow with the Web UI bundled in
  the Python package.

### Release status

This entry describes the intended first public alpha. It is not evidence that a
PyPI or GitHub release exists; the release is public only after the corresponding
tag and package artifacts are published.

[Unreleased]: https://github.com/5h1nnN/agentqakit/compare/v0.1.0a0...HEAD
[0.1.0a0]: https://github.com/5h1nnN/agentqakit/releases/tag/v0.1.0a0
