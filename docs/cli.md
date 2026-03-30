# Zernel CLI Reference

> Copyright (C) 2026 Dyber, Inc.

## Overview

The `zernel` CLI is a terminal-native development environment for ML/LLM workloads. It provides experiment tracking, model management, a live training dashboard, distributed job orchestration, and a SQL-like query language -- all from the terminal.

## Installation

```bash
cargo install --path zernel-cli

# Or from a release binary
# curl -fsSL https://get.zernel.dev | sh
```

## Commands

### `zernel init <name>`

Scaffold a new ML project.

```bash
zernel init my-project
```

Creates:
```
my-project/
  zernel.toml   # Project configuration
  train.py      # Starter training script
  data/         # Dataset directory
  models/       # Model checkpoints
  configs/      # Training configs
  scripts/      # Utility scripts
```

### `zernel run <script> [args...]`

Run a training script with automatic experiment tracking.

```bash
zernel run train.py
zernel run train.py --epochs 10 --lr 0.001
```

What happens:
1. Creates a new experiment entry in the local SQLite store
2. Records the git commit hash (if in a git repo)
3. Launches the script via Python
4. Captures stdout/stderr in real time
5. Extracts metrics from output (loss, accuracy, lr, throughput, etc.)
6. Periodically saves metrics to the experiment store
7. On completion, records final status and duration

**Recognized metric patterns:**
- `loss: 1.234` or `loss=1.234`
- `accuracy: 0.95` or `acc=0.95`
- `grad_norm: 0.89`
- `learning_rate: 3e-4` or `lr=0.001`
- `throughput: 412` or `samples/s: 412`
- `epoch: 5` or `step: 4821`
- `perplexity: 12.3` or `ppl: 12.3`
- `eval_loss: 1.1`

### `zernel watch`

Full-screen terminal dashboard showing real-time training metrics.

```bash
zernel watch
```

Dashboard panels:
- **GPU Utilization**: per-device utilization bars with memory usage
- **Training Metrics**: loss, step progress, ETA
- **eBPF Telemetry**: CUDA launch latency, NCCL duration, DataLoader wait, PCIe bandwidth
- **Scheduler Phase**: current ML workload phase indicator

Keyboard shortcuts:
- `q` / `Esc` -- quit
- `r` -- reset demo state

> **Note**: Currently runs in demo mode with simulated data. Connect to a running zerneld instance for live telemetry (coming in Phase 4).

### `zernel exp list`

List all tracked experiments.

```bash
zernel exp list
zernel exp list --limit 50
```

Output:
```
ID                           Name                     Status      Loss   Acc    Duration
-----------------------------------------------------------------------------------------------
exp-20260330-142315-a1b2c3d4 llama3-finetune          Done      1.1870  0.8340 2h 14m
exp-20260330-120100-e5f6g7h8 llama3-baseline          Done      1.2341  0.8210 1h 42m
```

### `zernel exp show <id>`

Show full details of an experiment.

```bash
zernel exp show exp-20260330-142315-a1b2c3d4
```

### `zernel exp compare <a> <b>`

Diff hyperparameters and metrics between two experiments.

```bash
zernel exp compare exp-001 exp-002
```

Output:
```
Comparing: exp-001 vs exp-002
          llama3-baseline vs llama3-lr-sweep

Hyperparameters:
  learning_rate: 0.0001 -> 0.0003
  warmup_steps: 100 -> 200

Metrics:
  loss: 1.2341 -> 1.1870 (-3.8%)
  accuracy: 0.8210 -> 0.8340 (+1.6%)
```

### `zernel exp delete <id>`

Delete an experiment from the store.

```bash
zernel exp delete exp-001
```

### `zernel model save <path>`

Save a model checkpoint to the local registry.

```bash
zernel model save ./checkpoints/epoch-10 --name llama3-v1 --tag production
```

Saves: checkpoint files + git commit + metadata.

### `zernel model list`

List all registered models.

```bash
zernel model list
```

### `zernel model deploy <name:tag>`

Deploy a model for inference.

```bash
zernel model deploy llama3-v1:production --port 8080
```

### `zernel job submit <script>`

Submit a distributed training job.

```bash
zernel job submit train.py --gpus-per-node 8 --nodes 4 --framework pytorch --backend nccl
```

### `zernel doctor`

Diagnose the Zernel environment.

```bash
zernel doctor
```

Checks: OS, Python, NVIDIA driver, CUDA, PyTorch, PyTorch CUDA, Git, zerneld status, experiment DB.

### `zernel query "<ZQL>"`

Query experiments with ZQL (Zernel Query Language).

```bash
zernel query "SELECT name, loss, learning_rate FROM experiments WHERE loss < 1.5 ORDER BY loss ASC LIMIT 10"
```

## ZQL Syntax

ZQL is a SQL-like query language for the experiment store.

```sql
SELECT <columns>
FROM <table>
[WHERE <condition> [AND <condition>...]]
[ORDER BY <column> [ASC|DESC]]
[LIMIT <n>]
```

**Available tables:** `experiments`

**Available columns:** `id`, `name`, `status`, `loss`, `accuracy`, `learning_rate`, and any metric extracted during training.

**Operators:** `=`, `!=`, `<`, `>`, `<=`, `>=`

**Examples:**
```sql
-- All experiments sorted by loss
SELECT * FROM experiments ORDER BY loss ASC

-- Experiments better than baseline
SELECT name, loss FROM experiments WHERE loss < 1.5

-- Failed experiments
SELECT name, status FROM experiments WHERE status = 'Failed'
```

## Data Storage

All data is stored locally in `~/.zernel/`:

```
~/.zernel/
  experiments/
    experiments.db          # SQLite database
  models/
    registry.json           # Model registry index
    <model-name>/
      <tag>/                # Checkpoint files
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ZERNEL_LOG` | `zernel=warn` | Log level filter |

## Configuration

Project-level configuration lives in `zernel.toml`:

```toml
[project]
name = "my-project"
version = "0.1.0"

[training]
framework = "pytorch"
gpus = "auto"

[tracking]
enabled = true
auto_log = true
```
