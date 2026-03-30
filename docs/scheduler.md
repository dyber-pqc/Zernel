# sched_ext ML Scheduler

> Copyright (C) 2026 Dyber, Inc.

## Overview

Zernel ships a custom `sched_ext` CPU scheduler written in Rust, purpose-built for ML workloads. It leverages the `sched_ext` framework (merged into Linux 6.12) to load a fully custom, pluggable CPU scheduler into the kernel at runtime -- without forking the kernel.

## Why a Custom Scheduler Matters for ML

Standard Linux schedulers (CFS, EEVDF) are designed for general-purpose workloads. ML training on GPUs has a fundamentally different workload profile:

1. **Data loading phase**: Many CPU threads doing I/O, deserializing tensors, augmenting data
2. **Forward pass phase**: CPU threads are mostly idle waiting for GPU compute
3. **Backward pass**: GPU-heavy, CPU coordination overhead
4. **NCCL all-reduce**: Network-heavy, CPU threads doing MPI-style coordination
5. **Optimizer step**: Short CPU burst, should be scheduled with high priority

The standard scheduler has no concept of these phases. Zernel's scheduler does.

## Phase Detection

The scheduler maintains per-task state and classifies each ML process into one of five phases:

```rust
enum WorkloadPhase {
    DataLoading,     // High I/O, many threads, CPU-intensive
    GpuCompute,      // CPU idle, waiting on GPU
    NcclCollective,  // Network coordination for collectives
    OptimizerStep,   // Short CPU burst after GPU sync
    Unknown,         // Non-ML or unclassified
}
```

### Detection Heuristics

| Phase | Signals |
|-------|---------|
| DataLoading | `io_wait_fraction > 0.3` AND `gpu_utilization < 10%` |
| GpuCompute | `gpu_utilization > 80%` AND `cpu_burst_duration == 0` |
| NcclCollective | `nccl_active == true` AND `futex_wait_count > 0` |
| OptimizerStep | `cpu_burst < 5ms` AND `last_gpu_sync > 0` |

### Phase Stability

To prevent rapid phase flapping, the scheduler uses a stability tracker. A phase transition only commits after `phase_stability_count` consecutive identical classifications (default: 3).

## Scheduling Policy

| Phase | CPU Priority | Preemption | Latency Target | Rationale |
|-------|-------------|------------|---------------|-----------|
| DataLoading | +10 (high) | Aggressive | 100us | More CPU = faster pipeline = GPU never starves |
| GpuCompute | -5 (low) | Immediate yield | -- | CPU doing nothing useful, give it to data loaders |
| NcclCollective | +10 (high) | Low latency | 50us | Collectives are on the critical path |
| OptimizerStep | +10 (high) | Preemptive | 100us | Minimize GPU idle time between steps |
| Unknown | 0 (normal) | Standard | -- | CFS-equivalent behavior |

## NUMA-Aware CPU Selection

On multi-socket servers, the scheduler reads NUMA topology from `/sys/devices/system/node/` and GPU-to-NUMA mappings from PCI sysfs:

```
NUMA Node 0: CPUs 0-31, GPU 0-3
NUMA Node 1: CPUs 32-63, GPU 4-7
```

When scheduling a task bound to GPU 2, the scheduler prefers CPUs 0-31 (same NUMA node). Within that set, it picks the CPU with the lowest current load.

## Multi-Tenant Scheduling

On shared GPU servers, the scheduler enforces GPU-proportional CPU allocation:

- If Job A has 2 GPUs and Job B has 6 GPUs, Job B gets 3x the CPU scheduling weight
- Priority classes: `Training > Inference > Interactive > Background`
- One job's data loading burst cannot starve another job's NCCL collectives

### Priority Classes

| Class | Base Priority | Use Case |
|-------|--------------|----------|
| Training | +5 | Training jobs -- highest resource allocation |
| Inference | +3 | Inference serving -- latency-sensitive |
| Interactive | +1 | Jupyter notebooks -- responsive but best-effort |
| Background | -5 | Batch preprocessing -- lowest priority |

## Configuration

The scheduler reads from `/etc/zernel/scheduler.toml`:

```toml
[general]
phase_eval_interval_ms = 100
gpu_poll_interval_ms = 500
max_tracked_tasks = 65536

[phase_detection]
io_wait_threshold = 0.3
optimizer_burst_max_ns = 5000000
gpu_idle_threshold = 10
gpu_active_threshold = 80
phase_stability_count = 3
nccl_detection_enabled = false

[numa]
enabled = true
gpu_affinity = true
memory_affinity = true

[multi_tenant]
enabled = false
default_priority_class = "normal"

[telemetry]
metrics_port = 9093
push_interval_ms = 1000
```

Generate a default config:

```bash
zernel-scheduler --dump-config > /etc/zernel/scheduler.toml
```

## Telemetry

The scheduler exports metrics in Prometheus format:

```
zernel_scheduler_tasks{type="total"} 42
zernel_scheduler_tasks{type="ml"} 8
zernel_scheduler_decisions_total 1847293
zernel_scheduler_phase_transitions_total 4821
zernel_scheduler_phase_tasks{phase="data_loading"} 3
zernel_scheduler_phase_tasks{phase="gpu_compute"} 4
zernel_scheduler_phase_time_pct{phase="gpu_compute"} 72.30
zernel_scheduler_phase_time_pct{phase="data_loading"} 18.50
```

## BPF Implementation

The scheduler uses two components:

1. **Userspace daemon** (Rust): Maintains per-task state, runs phase detection, writes scheduling decisions to BPF maps
2. **Kernel BPF program** (C): Implements `sched_ext` ops, reads decisions from BPF maps, applies them at scheduling decision points

The BPF program implements these hooks:
- `ops.select_cpu()` -- NUMA-aware CPU selection
- `ops.enqueue()` -- priority-based enqueue ordering
- `ops.dispatch()` -- phase-aware dispatch
- `ops.running()` / `ops.stopping()` -- phase timing

> **Note**: Full BPF implementation requires the `bpf` Cargo feature and Linux 6.12+. The userspace logic works on any platform for development and testing.
