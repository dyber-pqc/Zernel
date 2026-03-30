# eBPF Observability Layer

> Copyright (C) 2026 Dyber, Inc.

## Overview

`zerneld` is the Zernel observability daemon. It loads eBPF probes into the kernel to instrument ML workloads in real time -- without any application code changes. Think `perf` but purpose-built for ML.

## Probe Architecture

```
+-------------------+    +-------------------+    +-------------------+
| gpu_mem.bpf.c     |    | cuda_trace.bpf.c  |    | nccl.bpf.c        |
| uprobe: libcuda   |    | uprobe: libcuda   |    | uprobe: libnccl   |
+--------+----------+    +--------+----------+    +--------+----------+
         |                         |                         |
         v                         v                         v
+--------+-------------------------+-------------------------+--------+
|                    BPF Ring Buffers                                   |
+--------+-------------------------+-------------------------+--------+
         |                         |                         |
         v                         v                         v
+--------+----------+    +--------+----------+    +--------+----------+
| GpuMemConsumer    |    | CudaTraceConsumer |    | NcclConsumer      |
+--------+----------+    +--------+----------+    +--------+----------+
         |                         |                         |
         +------------+------------+------------+------------+
                      |                         |
                      v                         v
              +-------+--------+    +-----------+-----------+
              | AggregatedMetrics|    | AlertEngine          |
              +-------+--------+    +-----------+-----------+
                      |
         +------------+------------+
         |                         |
         v                         v
+--------+----------+    +--------+----------+
| Prometheus HTTP   |    | WebSocket Server  |
| :9091/metrics     |    | :9092             |
+-------------------+    +-------------------+
         |                         |
         v                         v
  Grafana / Alertmanager     zernel watch (CLI)
```

## What Gets Instrumented

### GPU Memory (`zernel-gpumem`)
- Tracks CUDA memory allocation/deallocation via uprobes on `libcuda.so`
- Detects OOM conditions before they happen
- Reports per-process GPU memory high watermarks
- Detects memory fragmentation patterns

### CUDA Kernel Launch Latency (`zernel-cuda-trace`)
- Instruments `cuLaunchKernel` / `cudaLaunchKernel` via uprobes
- Measures time from Python call to actual kernel execution
- Identifies PCIe transfer bottlenecks
- Reports per-operation latency histograms (p50, p99)

### NCCL Collective Bottlenecks (`zernel-nccl`)
- Instruments `ncclAllReduce`, `ncclBroadcast`, etc.
- Measures collective duration across all ranks
- Identifies straggler GPUs/nodes
- Tracks ring buffer utilization

### Dataset I/O Pipeline (`zernel-dataload`)
- Instruments `io_uring` and `read` syscalls from DataLoader workers
- Measures dataset prefetch efficiency
- Identifies storage vs CPU preprocessing bottlenecks
- Tracks cache hit/miss for dataset shards

### Distributed Synchronization (`zernel-dist`)
- Instruments `futex` and `pthread_barrier` calls
- Measures rank synchronization overhead
- Detects gradient accumulation stragglers

## Prometheus Metrics Reference

All metrics are exposed at `http://localhost:9091/metrics`.

```
# GPU Memory
zernel_gpu_memory_used_bytes{pid, gpu_id}
zernel_gpu_memory_peak_bytes{pid, gpu_id}

# CUDA Kernel Latency
zernel_cuda_launch_latency_seconds{pid, quantile="0.5"}
zernel_cuda_launch_latency_seconds{pid, quantile="0.99"}
zernel_cuda_launch_latency_seconds_count{pid}

# NCCL Collectives
zernel_nccl_collective_duration_seconds{op, quantile="0.5"}
zernel_nccl_collective_duration_seconds{op, quantile="0.99"}

# DataLoader
zernel_dataloader_wait_seconds{pid, quantile="0.5"}
zernel_dataloader_wait_seconds{pid, quantile="0.99"}
```

## WebSocket Protocol

zerneld pushes JSON snapshots to connected clients at a configurable interval (default: 1s):

```json
{
  "gpu_utilization": [
    {"key": "1000:0", "current_bytes": 83886080000, "peak_bytes": 83886080000}
  ],
  "cuda_latency_p50_us": 142.0,
  "cuda_latency_p99_us": 891.0,
  "nccl_allreduce_p50_ms": 34.0,
  "nccl_allreduce_p99_ms": 67.0,
  "dataloader_wait_p50_ms": 8.0,
  "last_update_ms": 1711800000000
}
```

Connect with any WebSocket client:

```bash
websocat ws://localhost:9092
```

## Running zerneld

```bash
# Development mode (simulated telemetry, no BPF)
zerneld --simulate

# Production mode (requires Linux + root)
sudo zerneld

# Custom ports
ZERNEL_LOG=debug zerneld --simulate
```

## Grafana Integration

Import the Zernel dashboard from `docs/grafana-dashboard.json` (coming soon) or create a Prometheus data source pointing to `http://zernel-host:9091`.

## Alert Configuration

Alerts are configured in the zerneld source. The default alert:

```
GPU OOM Warning: triggers when gpu_memory_used_pct > 95%
```

Custom alerts and webhook integrations are planned for Phase 5.
