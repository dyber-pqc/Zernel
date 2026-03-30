# Contributing to Zernel

Thank you for your interest in contributing to Zernel. This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, constructive, and professional. We are building infrastructure that ML teams depend on.

## Getting Started

### Development Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup component add rustfmt clippy

# Clone and build
git clone https://github.com/dyber-pqc/Zernel.git
cd Zernel
cargo build --workspace
cargo test --workspace
```

### Running the Full Check Suite

```bash
# Format check
cargo fmt --all -- --check

# Lints
cargo clippy --workspace -- -D warnings

# Tests
cargo test --workspace

# Or all at once:
./scripts/test-all.sh
```

## Project Structure

| Crate | License | Description |
|-------|---------|-------------|
| `zernel-scheduler` | GPL-2.0 | sched_ext ML-aware CPU scheduler |
| `zernel-ebpf` | GPL-2.0 | eBPF observability daemon |
| `zernel-cli` | Proprietary | Terminal-native CLI IDE |
| `distro/` | GPL-2.0 | Kernel config, sysctl, packages |
| `zernel-sdk/python` | MIT | Python SDK |

**Important**: Contributions to GPL-2.0 components are welcome from everyone. The CLI (proprietary) accepts contributions under a Contributor License Agreement (CLA).

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Ensure all checks pass:
   - `cargo fmt --all -- --check`
   - `cargo clippy --workspace -- -D warnings`
   - `cargo test --workspace`
5. Write clear commit messages
6. Submit a pull request

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- First line: concise summary (< 72 chars)
- Body: explain *why*, not just *what*

### Performance

Any PR that causes >2% throughput regression on the benchmark suite will be blocked from merge.

## Code Style

- **Rust**: Follow `rustfmt` defaults. Run `cargo fmt` before committing.
- **Copyright headers**: Every source file must include the appropriate copyright header:
  - GPL-2.0 components: `// Copyright (C) 2026 Dyber, Inc. -- GPL-2.0`
  - Proprietary components: `// Copyright (C) 2026 Dyber, Inc. -- Proprietary`
- **BPF C code**: Follow kernel coding style

## Reporting Issues

Open an issue on GitHub with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Hardware/OS information (for scheduler/eBPF issues)

## Areas for Contribution

- Scheduler phase detection heuristics and benchmarks
- eBPF probe implementations
- Additional metric extraction patterns in the CLI
- ZQL parser features
- Documentation improvements
- Hardware-specific testing and tuning

---

Copyright (C) 2026 Dyber, Inc.
