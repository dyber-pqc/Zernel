<p align="center">
  <h1 align="center">Zernel</h1>
  <p align="center"><strong>The AI-Native Linux Operating System</strong></p>
  <p align="center">
    A Linux distribution built from the ground up for machine learning and LLM workloads.<br>
    Custom CPU scheduler. Kernel-level observability. Terminal-native developer environment.
  </p>
  <p align="center">
    <a href="https://github.com/dyber-pqc/Zernel/actions"><img src="https://github.com/dyber-pqc/Zernel/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--2.0-blue.svg" alt="License: GPL-2.0"></a>
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.75%2B-orange.svg" alt="Rust 1.75+"></a>
    <a href="https://kernel.org/"><img src="https://img.shields.io/badge/linux-6.12%2B-yellow.svg" alt="Linux 6.12+"></a>
  </p>
</p>

---

## Why Zernel?

Every ML platform today is software you install *on top of* a general-purpose OS. Zernel **is** the OS, tuned from the kernel up for AI workloads.

Install Zernel on your GPU cluster. Everything you already run still works -- PyTorch, JAX, vLLM, Kubernetes -- but the OS itself is tuned for AI at the kernel level.

### The Problem

- Stock Linux schedulers (CFS/EEVDF) waste CPU cycles during GPU compute phases and starve data pipelines during loading phases
- ML teams spend weeks hand-tuning kernel parameters, NUMA policies, and huge page allocations
- GPU memory pressure, CUDA launch latency, and NCCL bottlenecks are invisible without invasive profiling
- There is no unified tool for experiment tracking, model management, and distributed job orchestration that works from the terminal

### The Solution

| Layer | Component | What It Does |
|-------|-----------|-------------|
| **5** | [Zernel CLI IDE](docs/cli.md) | Terminal-native experiment tracking, model registry, live dashboard, ZQL query language |
| **4** | [eBPF Observability](docs/ebpf.md) | Zero-instrumentation GPU memory, CUDA latency, NCCL, and DataLoader profiling |
| **3** | [sched_ext ML Scheduler](docs/scheduler.md) | Rust-based CPU scheduler that detects ML workload phases and schedules accordingly |
| **2** | [Kernel Configuration](docs/kernel-config.md) | Pre-tuned huge pages, NUMA policies, RDMA networking, ML-optimized sysctl |
| **1** | Linux 6.12+ / NVIDIA Open Drivers | Full CUDA, cuDNN, TensorRT, NCCL -- zero porting required |

---

## Quick Start

### Build from Source

```bash
# Prerequisites: Rust 1.75+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/dyber-pqc/Zernel.git
cd Zernel
cargo build --workspace --release

# Run tests (49 tests)
cargo test --workspace

# Install the CLI
cargo install --path zernel-cli
```

### Your First Run

```bash
# Initialize a new ML project
zernel init my-project
cd my-project

# Run a training script with automatic experiment tracking
zernel run train.py

# View the live dashboard
zernel watch

# List tracked experiments
zernel exp list

# Compare two experiments
zernel exp compare exp-001 exp-002

# Query experiments with ZQL
zernel query "SELECT name, loss FROM experiments WHERE loss < 1.5 ORDER BY loss ASC"

# Diagnose your environment
zernel doctor
```

### Start the Observability Daemon

```bash
# Start zerneld with simulated telemetry (development mode)
cargo run -p zernel-ebpf --release -- --simulate

# Prometheus metrics available at http://localhost:9091/metrics
# WebSocket telemetry stream at ws://localhost:9092
```

---

## Architecture

```
+---------------------------------------------------------------------+
|  LAYER 5: Zernel CLI IDE + Developer Environment                     |
|  Terminal-native IDE . Experiment tracking . Model versioning        |
|  Job orchestration . One-command deploy . ZQL query language         |
+---------------------------------------------------------------------+
|  LAYER 4: eBPF Observability Layer                                   |
|  GPU memory telemetry . CUDA launch profiling . NCCL bottlenecks     |
|  Real-time dashboards . Anomaly detection . No code changes needed   |
+---------------------------------------------------------------------+
|  LAYER 3: sched_ext ML Scheduler (Rust)                              |
|  Custom CPU scheduler . ML workload phase detection                  |
|  GPU-burst aware scheduling . Multi-tenant GPU server support        |
+---------------------------------------------------------------------+
|  LAYER 2: Zernel Kernel Configuration + Tuning                       |
|  Huge page pre-allocation . NUMA policies . RDMA networking          |
|  ML-optimized kernel parameters . Custom kernel .config              |
+---------------------------------------------------------------------+
|  LAYER 1: Linux Kernel (6.12+) + NVIDIA Open Drivers                 |
|  Full CUDA . cuDNN . TensorRT . NCCL . All ML frameworks             |
+---------------------------------------------------------------------+
```

---

## ML Scheduler: How It Works

Zernel's `sched_ext` scheduler understands ML workload phases and applies different CPU scheduling policies to each:

| Phase | Detection | CPU Policy | Why |
|-------|-----------|-----------|-----|
| **Data Loading** | High I/O wait + low GPU util | High priority, aggressive preemption | Faster data pipeline = GPU never starves |
| **GPU Compute** | High GPU util + CPU idle | Very low priority, immediate yield | CPU doing nothing useful -- give it to data loaders |
| **NCCL Collective** | NCCL shared memory + futex activity | High priority, <50us latency target | Collective ops are on the critical path |
| **Optimizer Step** | Short CPU burst after GPU sync | High priority, preemptive | Minimize GPU idle time between steps |

The scheduler also provides:
- **NUMA-aware CPU selection** -- tasks are pinned to CPUs on the same NUMA node as their GPU
- **Multi-tenant scheduling** -- GPU-proportional CPU allocation across jobs sharing a server
- **Phase stability tracking** -- debounced transitions to prevent scheduling flap

See [docs/scheduler.md](docs/scheduler.md) for the full technical deep-dive.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `zernel init <name>` | Scaffold a new ML project |
| `zernel run <script>` | Run with automatic GPU detection, metric extraction, experiment tracking |
| `zernel watch` | Full-screen TUI dashboard: GPU util, training metrics, eBPF telemetry |
| `zernel exp list` | List all tracked experiments |
| `zernel exp show <id>` | Show experiment details, hyperparameters, and metrics |
| `zernel exp compare <a> <b>` | Diff hyperparameters and metrics between two experiments |
| `zernel model save <path>` | Save a model checkpoint + metadata to the local registry |
| `zernel model list` | List all registered models |
| `zernel model deploy <name>` | Deploy a model for inference |
| `zernel job submit <script>` | Submit a distributed training job |
| `zernel doctor` | Diagnose GPU drivers, CUDA, PyTorch, zerneld, kernel config |
| `zernel query "<ZQL>"` | Query experiments with SQL-like ZQL syntax |

---

## Repository Structure

```
zernel/
|-- zernel-scheduler/          # Layer 3: sched_ext ML-aware CPU scheduler (Rust + BPF)
|-- zernel-ebpf/               # Layer 4: eBPF observability daemon (zerneld)
|-- zernel-cli/                # Layer 5: Terminal-native CLI IDE
|-- distro/                    # Layers 1-2: kernel config, sysctl, packages, installer
|   |-- kernel/config/         # ML-optimized kernel .config
|   |-- sysctl/                # Tuned sysctl parameters
|   |-- packages/              # Base + NVIDIA package lists
|   `-- installer/             # ISO builder and installer scripts
|-- zernel-sdk/                # Python and Rust SDKs
|-- docs/                      # Documentation
|-- tests/                     # Integration and benchmark tests
`-- scripts/                   # Build, test, and dev environment scripts
```

---

## Requirements

| Component | Requirement | Notes |
|-----------|------------|-------|
| Rust | 1.75+ stable | Build all crates |
| Linux | 6.12+ | Runtime only -- sched_ext and BPF |
| clang/llvm | 14+ | BPF compilation (Linux only) |
| NVIDIA GPU | Compute Capability 7.0+ | A100, H100, RTX 3090/4090+ |
| CUDA | 12.x | Pre-installed in Zernel distro |
| Python | 3.10+ | For ML framework stack |

> **Cross-platform development**: BPF compilation is feature-gated. `cargo build` and `cargo test` work on macOS and Windows for development. Full BPF features require Linux.

---

## Development

```bash
# Set up development environment
./scripts/setup-dev-env.sh

# Build all components
cargo build --workspace

# Run the test suite (49 tests)
cargo test --workspace

# Run lints
cargo clippy --workspace -- -D warnings

# Format check
cargo fmt --all -- --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/architecture.md) | System design, layer interactions, data flow |
| [Installation Guide](docs/installation.md) | Building from source, distro installation, hardware requirements |
| [Scheduler Deep-Dive](docs/scheduler.md) | Phase detection, NUMA awareness, multi-tenant scheduling |
| [eBPF Observability](docs/ebpf.md) | Probe architecture, metrics reference, Prometheus integration |
| [CLI Reference](docs/cli.md) | All commands, flags, configuration, ZQL syntax |
| [Kernel Configuration](docs/kernel-config.md) | Kernel .config, sysctl tuning, huge pages, RDMA |
| [Configuration Reference](docs/configuration.md) | All config files, environment variables, defaults |
| [API Reference](docs/api.md) | Prometheus metrics, WebSocket protocol, Python SDK |

---

## Technology Stack

| Component | Language | Key Libraries |
|-----------|---------|---------------|
| ML Scheduler | Rust + BPF C | libbpf-rs, scx_utils |
| zerneld (eBPF daemon) | Rust | libbpf-rs, hyper, tokio-tungstenite |
| Zernel CLI IDE | Rust | ratatui, clap, rusqlite, nom |
| ZQL Parser | Rust | nom |
| Python SDK | Python | -- |
| Kernel Config | -- | Linux 6.12+ |
| Distro Base | -- | Debian stable |
| NVIDIA Stack | -- | CUDA 12.x, cuDNN 9, TensorRT |

---

## License

Zernel uses an **open-core licensing model**:

- **Kernel components** (scheduler, eBPF probes, kernel config): [GPL-2.0](LICENSE)
- **CLI IDE and enterprise features**: [Proprietary](LICENSE-ENTERPRISE)
- **Python SDK**: MIT

This follows the proven Red Hat model: open-source kernel work builds technical credibility, proprietary tooling and support drive revenue.

---

## Roadmap

- [x] **Phase 0** -- Monorepo structure, kernel config, build system
- [x] **Phase 1** -- Scheduler MVP with phase detection and NUMA awareness
- [x] **Phase 2** -- eBPF observability with Prometheus and WebSocket
- [x] **Phase 3** -- CLI IDE with experiment tracking, TUI dashboard, ZQL
- [ ] **Phase 4** -- Full distro integration, bootable ISO, distributed training
- [ ] **Phase 5** -- Enterprise features, multi-tenant dashboard, beta program

---

<p align="center">
  <strong>Zernel</strong> -- The OS that ML infrastructure engineers wish existed.<br>
  Copyright &copy; 2026 <a href="https://dyber.io">Dyber, Inc.</a>
</p>
