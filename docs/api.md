# API Reference

> Copyright (C) 2026 Dyber, Inc.

## Prometheus Metrics API

**Endpoint**: `GET http://localhost:9091/metrics`

Returns metrics in Prometheus text exposition format (v0.0.4).

### zerneld Metrics

```
# GPU Memory
zernel_gpu_memory_used_bytes{pid="<pid>", gpu_id="<id>"}
zernel_gpu_memory_peak_bytes{pid="<pid>", gpu_id="<id>"}

# CUDA Kernel Launch Latency
zernel_cuda_launch_latency_seconds{pid="<pid>", quantile="0.5"}
zernel_cuda_launch_latency_seconds{pid="<pid>", quantile="0.99"}
zernel_cuda_launch_latency_seconds_count{pid="<pid>"}

# NCCL Collective Duration
zernel_nccl_collective_duration_seconds{op="<op>", quantile="0.5"}
zernel_nccl_collective_duration_seconds{op="<op>", quantile="0.99"}

# DataLoader Wait
zernel_dataloader_wait_seconds{pid="<pid>", quantile="0.5"}
zernel_dataloader_wait_seconds{pid="<pid>", quantile="0.99"}
```

### Scheduler Metrics (Planned)

```
zernel_scheduler_tasks{type="total|ml"}
zernel_scheduler_decisions_total
zernel_scheduler_phase_transitions_total
zernel_scheduler_phase_tasks{phase="data_loading|gpu_compute|nccl_collective|optimizer_step|unknown"}
zernel_scheduler_phase_time_pct{phase="..."}
```

## Health Check

**Endpoint**: `GET http://localhost:9091/health`

Returns `200 OK` with body `ok` when zerneld is healthy.

## JSON Metrics

**Endpoint**: `GET http://localhost:9091/json`

Returns the full aggregated metrics as JSON. Useful for debugging.

## WebSocket Telemetry Stream

**Endpoint**: `ws://localhost:9092`

Pushes JSON snapshots at a configurable interval (default: 1 second).

### Message Format

```json
{
  "gpu_utilization": [
    {
      "key": "<pid>:<gpu_id>",
      "current_bytes": 83886080000,
      "peak_bytes": 83886080000
    }
  ],
  "cuda_latency_p50_us": 142.0,
  "cuda_latency_p99_us": 891.0,
  "nccl_allreduce_p50_ms": 34.0,
  "nccl_allreduce_p99_ms": 67.0,
  "dataloader_wait_p50_ms": 8.0,
  "last_update_ms": 1711800000000
}
```

### Client Example

```python
import asyncio
import websockets
import json

async def watch():
    async with websockets.connect("ws://localhost:9092") as ws:
        async for message in ws:
            data = json.loads(message)
            print(f"CUDA p50: {data['cuda_latency_p50_us']:.0f}us")

asyncio.run(watch())
```

## Python SDK (Planned)

```python
import zernel

# Track an experiment programmatically
with zernel.experiment("my-run") as exp:
    exp.log_param("lr", 0.001)
    for step in range(1000):
        loss = train_step()
        exp.log_metric("loss", loss, step=step)
```

Install: `pip install zernel`
