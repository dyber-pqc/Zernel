# Zernel Architecture

> Copyright (C) 2026 Dyber, Inc.

## Overview

Zernel is structured as five discrete layers. Each layer is independently valuable and can be shipped incrementally. Build order follows the layer numbers.

```
+---------------------------------------------------------------------+
|  LAYER 5: Zernel CLI IDE + Developer Environment                     |
|  zernel-cli crate | Proprietary license                             |
+---------------------------------------------------------------------+
|  LAYER 4: eBPF Observability Layer (zerneld)                         |
|  zernel-ebpf crate | GPL-2.0                                        |
+---------------------------------------------------------------------+
|  LAYER 3: sched_ext ML Scheduler                                     |
|  zernel-scheduler crate | GPL-2.0                                    |
+---------------------------------------------------------------------+
|  LAYER 2: Zernel Kernel Configuration + Tuning                       |
|  distro/ directory | GPL-2.0                                         |
+---------------------------------------------------------------------+
|  LAYER 1: Linux Kernel (6.12+) + NVIDIA Open Drivers                 |
|  Upstream, not owned by Zernel                                       |
+---------------------------------------------------------------------+
```

## Data Flow

```
Training Script (PyTorch/JAX)
       |
       v
+------+-------+     +------------------+
| zernel run   |---->| Experiment Store  |  (SQLite, ~/.zernel/)
| (CLI Layer5) |     | metrics, params   |
+------+-------+     +------------------+
       |
       | stdout metrics extraction
       v
+------+-------+
| zernel watch |     (Ratatui TUI dashboard)
| (CLI Layer5) |<----+
+------+-------+     |
                      | WebSocket (port 9092)
                      |
               +------+-------+
               | zerneld      |     (Layer 4 eBPF daemon)
               | (zernel-ebpf)|---> Prometheus /metrics (port 9091)
               +------+-------+
                      |
                      | BPF ring buffers (on Linux with root)
                      |
               +------+-------+
               | BPF Probes   |     (uprobes on libcuda, libnccl)
               | gpu_mem      |
               | cuda_trace   |
               | nccl         |
               | dataload     |
               | dist_sync    |
               +------+-------+
                      |
                      v
               +------+-------+
               | sched_ext    |     (Layer 3 ML scheduler)
               | zernel-sched |---> BPF maps (task state, decisions)
               +--------------+
                      |
                      v
               +--------------+
               | Linux Kernel |     (Layer 1+2)
               | 6.12+ w/     |
               | Zernel config|
               +--------------+
```

## Crate Dependencies

```
zernel-cli (Layer 5)
  |-- connects to zerneld via WebSocket
  |-- reads/writes SQLite experiment store
  `-- (no Cargo dependency on other Zernel crates)

zernel-ebpf (Layer 4)
  |-- loads BPF probes into kernel
  |-- serves Prometheus HTTP + WebSocket
  `-- (no Cargo dependency on other Zernel crates)

zernel-scheduler (Layer 3)
  |-- loads sched_ext BPF scheduler
  |-- maintains task state in BPF maps
  `-- (no Cargo dependency on other Zernel crates)
```

The three crates communicate at runtime via:
- **BPF maps** (scheduler <-> kernel)
- **WebSocket** (zerneld -> CLI)
- **Prometheus HTTP** (zerneld -> any monitoring stack)

They have no compile-time Cargo dependencies on each other, enabling independent builds and releases.

## Repository Layout

```
zernel/
|-- Cargo.toml                 # Workspace root
|-- zernel-scheduler/          # Layer 3
|   |-- src/
|   |   |-- main.rs            # Entry point, BPF loader
|   |   |-- scheduler.rs       # Core scheduling logic
|   |   |-- phase_detector.rs  # ML workload phase detection
|   |   |-- task_state.rs      # Per-task state + phase time tracking
|   |   |-- numa.rs            # NUMA topology detection + CPU selection
|   |   |-- multi_tenant.rs    # GPU-proportional multi-tenant scheduling
|   |   |-- config.rs          # TOML configuration system
|   |   |-- telemetry.rs       # Prometheus metrics export
|   |   `-- bpf/               # BPF C source (reference skeletons)
|   `-- Cargo.toml
|-- zernel-ebpf/               # Layer 4
|   |-- src/
|   |   |-- main.rs            # zerneld daemon entry point
|   |   |-- loader.rs          # BPF program loader
|   |   |-- aggregation.rs     # Metrics aggregation + histograms
|   |   |-- metrics_server.rs  # Prometheus HTTP server (hyper)
|   |   |-- websocket_server.rs# WebSocket server (tokio-tungstenite)
|   |   |-- simulator.rs       # Simulated telemetry for dev/demos
|   |   |-- alerts.rs          # Threshold-based alert engine
|   |   `-- consumers/         # Per-probe event consumers
|   `-- Cargo.toml
|-- zernel-cli/                # Layer 5
|   |-- src/
|   |   |-- main.rs            # CLI entry point (clap)
|   |   |-- commands/          # Subcommand implementations
|   |   |-- experiments/       # SQLite store, metric extraction, comparison
|   |   |-- telemetry/         # WebSocket client, display formatting
|   |   `-- zql/               # ZQL parser (nom), executor, schema
|   `-- Cargo.toml
|-- distro/                    # Layers 1-2
|   |-- kernel/config/         # ML-optimized kernel .config
|   |-- sysctl/                # /etc/sysctl.d/99-zernel.conf
|   |-- packages/              # Debian + NVIDIA package lists
|   `-- installer/             # ISO builder, installer script
|-- zernel-sdk/                # SDKs
|   `-- python/                # pip install zernel
|-- docs/                      # This directory
|-- scripts/                   # Build, test, dev setup
`-- .github/workflows/         # CI/CD
```

## Design Principles

1. **Layers are independent** -- each layer adds value without requiring the ones above it
2. **No application changes required** -- all observability is via eBPF, not SDK instrumentation
3. **Linux-native** -- builds on mainline kernel, not a fork. NVIDIA works day one.
4. **Open core** -- kernel work is GPL, proprietary value is in tooling and enterprise features
5. **Terminal-first** -- ML engineers live in the terminal. The CLI is the primary UX.
