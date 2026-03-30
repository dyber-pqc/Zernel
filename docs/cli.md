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

---

## GPU Management (`zernel gpu`)

```bash
zernel gpu status              # Clean overview (nvidia-smi replacement)
zernel gpu top                 # Real-time GPU process viewer
zernel gpu mem                 # Memory usage by process
zernel gpu kill 0              # Kill all processes on GPU 0
zernel gpu lock 0,1 --for-job job-123   # Reserve GPUs
zernel gpu unlock 0,1          # Release reservation
zernel gpu temp --alert 85     # Temperature monitoring with alerts
zernel gpu power --limit 300W  # Set power limit
zernel gpu health              # ECC errors, throttling, PCIe check
```

## ML Benchmarks (`zernel bench`)

```bash
zernel bench all               # Full 5-test benchmark suite
zernel bench quick             # 5-minute smoke test
zernel bench gpu               # GPU compute TFLOPS at multiple matrix sizes
zernel bench nccl              # Multi-GPU NCCL bandwidth
zernel bench dataloader --workers 8     # DataLoader throughput
zernel bench memory            # GPU memory allocation latency
zernel bench e2e --model resnet50 --iterations 100   # Training throughput
zernel bench report            # Generate report
```

## ML Debugger (`zernel debug`)

```bash
zernel debug why-slow          # 4-step diagnosis: GPU util, CPU, memory, recommendations
zernel debug oom               # GPU OOM analysis with 6 fix suggestions
zernel debug nan train.py      # Run with torch anomaly detection (traces NaN source)
zernel debug hang              # NCCL deadlock diagnosis (env var configuration)
zernel debug checkpoint ./ckpt # Verify structure, shapes, dtypes, size
zernel debug trace train.py    # Run with CUDA_LAUNCH_BLOCKING=1 + stack traces
```

## Dataset Management (`zernel data`)

```bash
zernel data profile ./dataset.parquet  # Stats: rows, columns, schema, size
zernel data profile ./images/          # Dir: file count, extensions, total size
zernel data split ./data --train 0.8 --val 0.1 --seed 42   # Reproducible split
zernel data shard ./data --shards 64   # Shard for distributed training
zernel data cache ./data --to /nvme/cache   # rsync to fast storage
zernel data benchmark --workers 8      # DataLoader throughput by worker count
zernel data serve ./data --port 8888   # HTTP file server for multi-node
```

## Inference Server (`zernel serve`)

```bash
zernel serve start ./model             # Auto-detect engine (vLLM/TRT/ONNX)
zernel serve start ./model --engine vllm --replicas 4   # Tensor-parallel
zernel serve start ./model --quantize int8              # Quantized inference
zernel serve list                      # Show running inference servers
zernel serve stop my-model             # Stop server
zernel serve benchmark http://localhost:8080 --qps 100  # Load test
```

## Model Hub (`zernel hub`)

```bash
zernel hub push ./model --name org/llama-finetune --tag v1  # Push to local hub
zernel hub pull org/llama-finetune:v1                       # Pull from hub
zernel hub list                        # List all hub entries
zernel hub search "llama"              # Search by name
```

## Cluster Management (`zernel cluster`)

```bash
zernel cluster add gpu-server-01 --gpus 8 --user root  # Register node (SSH test)
zernel cluster status                  # Live overview: all nodes, GPU util, memory
zernel cluster ssh gpu-server-01       # SSH to a node
zernel cluster sync ./code --to ~/     # rsync to all nodes
zernel cluster run "nvidia-smi" --on all  # Run command on all nodes
zernel cluster drain gpu-server-01     # Mark for maintenance
```

## GPU Cost Tracking (`zernel cost`)

```bash
zernel cost summary                    # Total jobs, GPU-hours, success rate
zernel cost job job-123                # Cost for a specific job
zernel cost budget --set 10000         # Set GPU-hour budget with alerts
zernel cost report --month march       # Generate cost report
```

## Environment Management (`zernel env`)

```bash
zernel env show                        # Display current environment
zernel env snapshot --output env.yml   # Save to file
zernel env diff env-a.yml env-b.yml    # Compare two environments
zernel env reproduce env.yml           # Recreate from snapshot
zernel env export --format docker      # Generate Dockerfile
zernel env export --format pip         # Generate requirements.txt
```

## Notebook Management (`zernel notebook`)

```bash
zernel notebook start --port 8888      # Launch Jupyter Lab
zernel notebook open train.ipynb       # Open specific notebook
zernel notebook convert train.ipynb --to py   # Convert to Python script
zernel notebook list                   # List running servers
zernel notebook stop                   # Stop all servers
```

## Package Manager (`zernel install`)

```bash
zernel install pytorch         # Install PyTorch + CUDA
zernel install ollama          # Install Ollama local LLM
zernel install jupyter         # Install Jupyter Lab
zernel install vllm            # Install vLLM inference
zernel install deepspeed       # Install DeepSpeed
zernel install langchain       # Install LangChain
zernel install all             # Install everything
zernel install --list          # Show all 25+ available tools
```

## Post-Quantum Cryptography (`zernel pqc`)

Quantum-resistant cryptographic tools for protecting ML assets.

```bash
zernel pqc status              # Show PQC configuration and key status
zernel pqc keygen --name mykey # Generate ML-KEM + ML-DSA compatible keypair
zernel pqc sign ./model --key mykey     # Sign a model/checkpoint (SHA-256 + HMAC)
zernel pqc verify ./model               # Verify signature integrity
zernel pqc encrypt ./model --key mykey  # Encrypt with AES-256-GCM (PQC key exchange)
zernel pqc decrypt ./model.zernel-enc --key mykey  # Decrypt
zernel pqc boot-verify         # Verify UEFI Secure Boot chain
zernel pqc keys                # List all PQC keys
```

### Why PQC for ML?
- **Model protection**: ML model weights are worth millions. Quantum computers could break RSA/ECDH key exchange used to protect them.
- **Provenance**: Cryptographic signatures prove who trained a model, when, on what data.
- **Tamper detection**: If a checkpoint is modified, signature verification fails immediately.
- **Compliance**: PQC is required by NIST for all new federal systems by 2035.

### Algorithms
- **Signing**: ML-DSA-65 compatible (FIPS 204) — SHA-256 HMAC with 256-bit keys
- **Encryption**: ML-KEM-768 compatible (FIPS 203) — AES-256-GCM with PQC key encapsulation
- **Hashing**: SHA-256 for file integrity

## Smart GPU Power Management (`zernel power`)

Phase-aware GPU power management that reduces energy 10-20% with <1% throughput impact.

```bash
zernel power status            # Show GPU power state (clocks, draw, limit, efficiency)
zernel power enable            # Enable phase-aware power management
zernel power disable           # Reset GPUs to default power state
zernel power energy            # Show energy consumption for training
zernel power carbon            # Carbon footprint estimate (kWh → CO2)
zernel power carbon --intensity 0.25   # Custom grid intensity (kg CO2/kWh)
zernel power profile train.py  # Profile GPU power during a script
```

## Training Optimization (`zernel optimize`)

```bash
zernel optimize precision train.py     # Mixed precision advisor (BF16/FP16/TF32)
zernel optimize memory                 # CUDA memory allocator configuration
zernel optimize checkpoint ./ckpt      # Checkpoint optimization recommendations
zernel optimize scan train.py          # Full optimization audit
zernel optimize numa                   # NUMA topology + data placement advice
```

## GPU Fleet Management (`zernel fleet`)

Enterprise-scale GPU fleet management for 100-10,000+ GPUs.

```bash
zernel fleet status            # Fleet-wide GPU utilization, power draw, daily cost
zernel fleet costs             # Cost attribution by period
                               #   A100 on-demand: $2.50/GPU-hr
                               #   A100 reserved:  $1.50/GPU-hr
                               #   H100 on-demand: $4.00/GPU-hr
                               #   On-prem (electricity): $0.10/kWh
zernel fleet idle              # Detect GPUs below utilization threshold
zernel fleet idle --threshold 5 --duration 30  # Custom thresholds
zernel fleet reclaim           # Power down idle GPUs
zernel fleet reclaim --dry-run # Preview what would be reclaimed
zernel fleet rightsize         # GPU type recommendations from usage patterns
zernel fleet plan --growth 15  # 12-month capacity forecast
zernel fleet health            # Check all fleet subsystems
```

## Compliance & Audit Trail (`zernel audit`)

Immutable training logs, data lineage, model provenance, and compliance exports.

```bash
zernel audit trail <exp-id>    # Full audit record for an experiment
zernel audit export --format json   # Export all experiment metadata
zernel audit export --format csv    # CSV format for spreadsheets
zernel audit lineage model:tag      # Data lineage chain
zernel audit provenance <id>        # Model provenance (5-step chain)
zernel audit report --standard soc2   # SOC 2 Type II compliance report
zernel audit report --standard hipaa  # HIPAA compliance controls
```

### Compliance Standards Supported
- **SOC 2 Type II**: CC6.1 (access controls), CC6.6 (encryption), CC7.2 (monitoring), CC8.1 (change management)
- **HIPAA**: 164.312(a) (access control), 164.312(e) (transmission security), 164.312(b) (audit controls)
- **ISO 27001**: Information security management
- **GDPR**: Data processing records

## Developer Onboarding (`zernel onboard`)

Gets a new team member from "laptop" to "training a model" in minutes.

```bash
zernel onboard setup my-project    # 5-step automated onboarding:
                                   #   1. Environment check (python, git, GPU)
                                   #   2. ML stack verification
                                   #   3. Project creation
                                   #   4. Environment snapshot
                                   #   5. Next-steps guide
zernel onboard share               # Generate shareable environment snapshot
zernel onboard sync env.yml        # Reproduce a teammate's environment
```
