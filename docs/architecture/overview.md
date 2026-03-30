# Zernel Architecture Overview

Zernel is structured as five discrete layers, each independently valuable.

## Layers

1. **Linux Base + NVIDIA Stack** — Mainline Linux 6.12+, full CUDA/cuDNN/TensorRT
2. **Kernel Configuration** — ML-optimized huge pages, NUMA, RDMA, sysctl tuning
3. **sched_ext ML Scheduler** — Rust-based CPU scheduler with ML workload phase detection
4. **eBPF Observability** — GPU memory, CUDA latency, NCCL profiling via zerneld
5. **CLI IDE** — Terminal-native experiment tracking, model registry, job orchestration

## Repository Layout

```
zernel/
├── zernel-scheduler/    # Layer 3: sched_ext scheduler (Rust + BPF)
├── zernel-ebpf/         # Layer 4: eBPF observability daemon
├── zernel-cli/          # Layer 5: CLI IDE
├── distro/              # Layers 1-2: kernel config, packages, installer
├── zernel-sdk/          # Python and Rust SDKs
├── docs/                # Documentation
├── tests/               # Integration and benchmark tests
└── scripts/             # Build and development scripts
```

Copyright (C) 2026 Dyber, Inc.
