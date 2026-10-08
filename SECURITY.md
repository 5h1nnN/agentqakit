# Security Policy

## Supported versions

AgentQAKit is currently pre-release software. Security fixes are applied to the
latest revision of the default branch until the first stable release defines a
formal support window.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities, credentials, or private test data in
a public issue. Use GitHub's private vulnerability reporting for this repository
when available. If that channel is unavailable, contact a maintainer privately
and include only the minimum information needed to reproduce the issue.

You should receive an acknowledgement within seven days. Maintainers will
coordinate validation, remediation, and disclosure timing with the reporter.

## Secrets and test artifacts

AgentQAKit artifacts must never contain raw authorization headers, cookies, API
keys, or other credentials. Use synthetic data in bug reports and reproduction
bundles.
