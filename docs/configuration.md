# Configuration Reference

> Copyright (C) 2026 Dyber, Inc.

## Overview

Zernel components are configured via TOML files and environment variables.

## Scheduler Configuration

**File**: `/etc/zernel/scheduler.toml`

Generate defaults: `zernel-scheduler --dump-config`

```toml
[general]
phase_eval_interval_ms = 100    # How often to re-evaluate task phases
gpu_poll_interval_ms = 500      # How often to poll GPU utilization
max_tracked_tasks = 65536       # Maximum tracked tasks
log_level = "info"              # Log level

[phase_detection]
io_wait_threshold = 0.3         # I/O wait fraction for DataLoading detection
optimizer_burst_max_ns = 5000000 # Max CPU burst (ns) for OptimizerStep (5ms)
gpu_idle_threshold = 10         # GPU util % below which GPU is "idle"
gpu_active_threshold = 80       # GPU util % above which GPU is "active"
phase_stability_count = 3       # Consecutive samples before phase transition
nccl_detection_enabled = false  # Enable NCCL collective detection

[numa]
enabled = true                  # Enable NUMA-aware CPU selection
gpu_affinity = true             # Prefer CPUs near task's GPU
memory_affinity = true          # Prefer CPUs near task's memory

[multi_tenant]
enabled = false                 # Enable multi-tenant scheduling
default_priority_class = "normal"

[telemetry]
metrics_port = 9093             # Prometheus metrics port (0 = disabled)
push_interval_ms = 1000         # Telemetry push interval
```

## zerneld Configuration

Currently configured via command-line flags and environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| Prometheus port | 9091 | `--metrics-port` (planned) |
| WebSocket port | 9092 | `--ws-port` (planned) |
| Push interval | 1000ms | `--push-interval` (planned) |
| Simulate mode | false | `--simulate` |

## CLI Configuration

### Project Config: `zernel.toml`

Created by `zernel init`, lives in the project root:

```toml
[project]
name = "my-project"
version = "0.1.0"

[training]
framework = "pytorch"   # pytorch, jax
gpus = "auto"            # auto, 0, "0,1,2,3"

[tracking]
enabled = true           # Enable experiment tracking
auto_log = true          # Auto-extract metrics from stdout
```

### Data Directory: `~/.zernel/`

```
~/.zernel/
  experiments/
    experiments.db      # SQLite experiment database
  models/
    registry.json       # Model registry index
    <model>/
      <tag>/            # Checkpoint files
```

## Environment Variables

| Variable | Default | Used By | Description |
|----------|---------|---------|-------------|
| `ZERNEL_LOG` | varies | All | Log level filter (e.g., `zernel=debug`) |
| `ZERNEL_SCHEDULER_CONFIG` | `/etc/zernel/scheduler.toml` | Scheduler | Config file path |

## Kernel sysctl

**File**: `/etc/sysctl.d/99-zernel.conf`

See [kernel-config.md](kernel-config.md) for the full list of tuned parameters.

## Port Reference

| Port | Service | Protocol |
|------|---------|----------|
| 9091 | zerneld Prometheus metrics | HTTP |
| 9092 | zerneld WebSocket telemetry | WebSocket |
| 9093 | Scheduler metrics (planned) | HTTP |
