# Installation Guide

> Copyright (C) 2026 Dyber, Inc.

## Building from Source

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | 1.75+ stable | Build all Zernel crates |
| Python | 3.10+ | ML framework stack |
| Git | 2.x+ | Version control, experiment tracking |

**Linux-only** (for full BPF features):
| Tool | Version | Purpose |
|------|---------|---------|
| clang/llvm | 14+ | BPF program compilation |
| libelf-dev | -- | ELF parsing for BPF |
| linux-headers | 6.12+ | Kernel headers for BPF |
| bpftool | -- | BPF program management |

### Build Steps

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Clone the repository
git clone https://github.com/dyber-pqc/Zernel.git
cd Zernel

# Build all components (release mode)
cargo build --workspace --release

# Verify the build
cargo test --workspace
```

### Install the CLI

```bash
# Install the zernel binary to ~/.cargo/bin/
cargo install --path zernel-cli

# Verify
zernel --version
zernel doctor
```

### Install zerneld (Observability Daemon)

```bash
# Copy binary to system path
sudo cp target/release/zernel-ebpf /usr/local/bin/zerneld

# Start in simulation mode (no BPF probes)
zerneld --simulate

# Or with BPF probes (requires Linux + root)
sudo zerneld
```

### Install the Scheduler

```bash
# Copy binary
sudo cp target/release/zernel-scheduler /usr/local/bin/zernel-scheduler

# Create config directory
sudo mkdir -p /etc/zernel

# Generate default config
zernel-scheduler --dump-config | sudo tee /etc/zernel/scheduler.toml

# Start (requires Linux 6.12+ with BPF feature)
sudo zernel-scheduler
```

## Cross-Platform Development

BPF compilation is gated behind a Cargo feature flag. On macOS and Windows, the crates build and test in userspace-only mode:

```bash
# Works on macOS, Windows, Linux
cargo build --workspace
cargo test --workspace

# Linux-only: build with BPF support
cargo build --workspace --features bpf
```

## Hardware Requirements

### Minimum (Development)

- Any x86_64 CPU
- 8 GB RAM
- Any NVIDIA GPU (for testing CUDA detection)

### Recommended (Production)

| Component | Specification |
|-----------|--------------|
| CPU | AMD EPYC or Intel Xeon (dual socket for NUMA testing) |
| RAM | 256 GB+ |
| GPU | NVIDIA A100 80GB SXM or H100 NVLink |
| Network | Mellanox InfiniBand HDR or 100GbE RoCE |
| Storage | NVMe SSD (for dataset I/O) |

### Tested Hardware

| Hardware | Status |
|----------|--------|
| NVIDIA A100 80GB SXM | Full test suite |
| NVIDIA H100 NVLink | Full test suite |
| NVIDIA RTX 4090 | Basic tests |
| AMD EPYC (dual socket) | NUMA tests |

## Zernel Distro Installation

> **Note**: The full Zernel ISO is under development (Phase 4). For now, use the component-level installation above on an existing Linux system.

When available, the ISO will be installable via:

```bash
# Boot from Zernel ISO, then:
zernel-install

# Or unattended:
zernel-install --config zernel-install-unattended.yaml
```

The installer will automatically:
1. Partition disks
2. Install the Zernel kernel with ML-optimized config
3. Install NVIDIA drivers + CUDA toolkit
4. Install zernel-scheduler, zerneld, and zernel CLI
5. Apply sysctl tuning
6. Install the ML framework stack (PyTorch, JAX, vLLM, etc.)
