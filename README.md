# Zernel

**AI-Native Linux OS for Machine Learning**

Zernel is a Linux distribution built from the ground up for ML and LLM workloads. It ships a custom CPU scheduler, kernel-level observability, and a terminal-native developer environment — all tuned for GPU-accelerated training and inference.

## Architecture

```
Layer 5  Zernel CLI IDE — experiment tracking, model registry, job orchestration
Layer 4  eBPF Observability — GPU memory, CUDA latency, NCCL profiling
Layer 3  sched_ext ML Scheduler — Rust-based, ML workload phase-aware
Layer 2  Kernel Configuration — huge pages, NUMA, RDMA, ML-optimized params
Layer 1  Linux 6.12+ / NVIDIA Open Drivers — full CUDA/cuDNN/TensorRT
```

## Quick Start

```bash
# Build all components
./scripts/build-all.sh

# Run tests
./scripts/test-all.sh
```

## Components

| Crate | Description |
|---|---|
| `zernel-scheduler` | sched_ext ML-aware CPU scheduler (Rust + BPF) |
| `zernel-ebpf` | eBPF observability daemon (zerneld) |
| `zernel-cli` | Terminal-native CLI IDE |

## Requirements

- Rust 1.75+ (stable)
- Linux 6.12+ (for sched_ext and BPF — runtime only)
- clang/llvm (for BPF compilation — Linux only)
- NVIDIA GPU + CUDA 12.x (runtime only)

## License

- Kernel components (scheduler, eBPF): GPL v2
- CLI IDE and enterprise features: Proprietary (see LICENSE-ENTERPRISE)

Copyright (C) 2026 Dyber, Inc.
