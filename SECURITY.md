# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please report security issues by emailing:

**security@dyber.io**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Timeline

- **48 hours**: Acknowledgment of your report
- **7 days**: Initial assessment and severity classification
- **90 days**: Fix developed, tested, and released

## Scope

The following components are in scope:
- `zernel-cli` — CLI IDE binary
- `zernel-ebpf` — zerneld observability daemon
- `zernel-scheduler` — sched_ext CPU scheduler
- `distro/` — kernel configuration and sysctl tuning

## Out of Scope

- Vulnerabilities in upstream dependencies (report to the respective project)
- Issues requiring physical access to the machine
- Social engineering attacks

## Credit

We will credit security researchers in the release notes (with your permission).

---

Copyright (C) 2026 Dyber, Inc.
