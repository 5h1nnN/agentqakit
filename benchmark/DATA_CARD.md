# AgentQA-Bench data card

## Summary

AgentQA-Bench 1.0.0 is a deterministic benchmark for testing tool-using AI agents. It
contains 110 declarative AgentTest cases and 35 mutation specifications across two small
LangGraph subjects. Twenty cases are marked as Gold Cases for end-to-end calibration,
including optional LLM Judge evaluation.

All people, courses, grades, issues, repositories, credentials, and documents are
synthetic. The benchmark contains no collected conversations or personal data.

## Composition

| Category | Cases |
|---|---:|
| Normal tasks | 30 |
| Boundary inputs | 15 |
| Tool calls and arguments | 15 |
| Tool failure recovery | 15 |
| Retrieval and knowledge faults | 10 |
| Permission and privacy | 15 |
| Prompt injection and security | 10 |
| **Total** | **110** |

The authoritative counts and benchmark SHA-256 are generated into `manifest.json` by
`scripts/generate_benchmark.py`. Do not edit generated YAML, `VERSION`, or the manifest by
hand.

## Collection and annotation

Cases are generated from reviewed, deterministic templates. Every case records:

- category and target agent;
- synthetic source provenance;
- risk level;
- Apache-2.0 license label;
- Gold Case status;
- deterministic expectations and, where applicable, an LLM Judge rubric.

High-risk permission, privacy, and prompt-injection cases always contain deterministic
assertions. They never rely only on an LLM Judge.

## Intended use

Use the benchmark to compare output-only evaluation, semantic judging, trace rules,
mutation testing, and record/replay. It is designed for regression experiments and
teaching. It does not estimate real-world safety, fairness, or model capability.

## Known limitations

- The agents are compact rule-based LangGraph programs rather than production LLM agents.
- Synthetic data does not reproduce every ambiguity of real campus or repository data.
- LLM Judge outcomes depend on the configured provider and model; preserve raw outputs and
  environment metadata when reporting results.
- A benchmark pass is evidence about these cases only, not a security guarantee.

## License and maintenance

The dataset and generator are licensed under Apache-2.0 with the rest of AgentQAKit.
Contributions must keep cases synthetic, document their source, and regenerate the fixed
hash. Security-sensitive additions require human review.

