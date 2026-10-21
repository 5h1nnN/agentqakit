# Releasing AgentQAKit

This runbook prepares one reproducible Python distribution containing the SDK,
CLI, API, MCP integration, and built React UI. It does not create a separate SDK
package.

## Release prerequisites

- The release commit is on `main` and CI is green.
- `pyproject.toml` contains the intended PEP 440 version.
- `CHANGELOG.md` has a dated entry for that exact version.
- The repository contains no `.env.local`, credentials, runtime databases, or
  unreviewed artifacts in Git.
- PyPI and TestPyPI Trusted Publishers point only to
  `.github/workflows/release.yml` in `5h1nnN/agentqakit`.
- The GitHub environments `testpypi` and `pypi` exist; `pypi` requires a human
  approver.

## Local release check

Use Python 3.11 or 3.12 and Node.js 20.19 or newer:

```bash
python -m pip install -e ".[dev,langgraph,mcp]"
make release-check
```

The check runs linting, typing, Python tests, schema verification, leak scanning,
Web tests and build, then creates one wheel and one source distribution. The Web
build is copied into `agentqa/_static/dist` only for packaging and is ignored by
Git. `scripts/check_release.py` verifies that both artifacts contain the UI.

Inspect the artifacts and test the wheel in a clean environment:

```bash
python -m venv .release-venv
.release-venv/bin/python -m pip install "dist/agentqakit-<version>-py3-none-any.whl[web]"
.release-venv/bin/agentqa --version
.release-venv/bin/python -c "from agentqa import agent, tool, retriever"
```

On Windows, use `.release-venv\Scripts\python.exe` and
`.release-venv\Scripts\agentqa.exe`.

## Trusted Publishing workflow

1. Run the Release workflow with `target=build-only` from the intended commit.
2. Download and inspect the `python-package-distributions` artifact.
3. Create the signed or annotated tag `v<version>` at the verified commit and
   push it.
4. Run the workflow on that tag with `target=testpypi`.
5. Install the exact version from TestPyPI in a clean environment, using PyPI
   only as the dependency fallback, and repeat the CLI, SDK, and Web smoke tests:

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ "agentqakit[web]==<version>"
   ```

6. Run the workflow again on the same tag with `target=pypi` and approve the
   protected `pypi` environment.
7. Create a GitHub Release from the tag, attach the wheel and sdist, and paste
   the matching changelog entry.

PyPI files and public version identifiers are immutable. If a release is wrong,
increment the version; never replace an uploaded artifact.

## Post-release verification

```bash
python -m pip install "agentqakit[web]==<version>"
agentqa --version
agentqa serve --workspace /path/to/a/test/project
```

Confirm that the workbench loads from the installed package, then verify the
SDK example, project links, license metadata, checksums, and GitHub Release.
Update the technical report only after these public URLs are live.
