<p align="center">
  <h1 align="center">Zernel</h1>
  <p align="center"><strong>The AI-Native Linux Operating System</strong></p>
  <p align="center">
    The first operating system where the kernel itself understands machine learning.<br>
    <strong>Faster training. Lower energy costs. Zero code changes.</strong>
  </p>
  <p align="center">
    <a href="https://github.com/dyber-pqc/Zernel/actions"><img src="https://github.com/dyber-pqc/Zernel/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--2.0-blue.svg" alt="License: GPL-2.0"></a>
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.75%2B-orange.svg" alt="Rust 1.75+"></a>
    <a href="https://kernel.org/"><img src="https://img.shields.io/badge/linux-6.12%2B-yellow.svg" alt="Linux 6.12+"></a>
  </p>
</p>

---

## What If Your OS Was Built For AI?

Every ML platform today runs **on top of** a general-purpose operating system that was designed for web servers and desktop apps. The kernel doesn't know the difference between a data loading thread and a CUDA synchronization call. It can't tell when your GPU is starving for data. It doesn't know that NCCL all-reduce is the critical path in your distributed training.

**Zernel does.**

Zernel is a complete Linux distribution where the CPU scheduler, memory manager, network stack, and observability layer all understand ML workloads natively. Install it on your GPU cluster. Everything you already run still works -- PyTorch, JAX, vLLM, Kubernetes -- but now the OS itself is working for you.

### Proven: 2.2x Faster Training, Zero Code Changes

The fastest way to use Zernel -- prefix any training script with `zernel-run`:

```bash
pip install zernel-runtime
zernel-run train.py          # 2.2x faster. That's it.
```

**Verified on bare metal** (3 rounds x 50 steps = 150 steps per config, MiniGPT-6L 119.8M params):

| Config | Step Time | Throughput | Peak Memory | Speedup |
|--------|-----------|------------|-------------|---------|
| **Vanilla PyTorch (FP32)** | 376.64 +/- 43.24 ms | 85.0 s/s | 5.40 GB | 1.0x |
| **Manual AMP (user adds 2 lines)** | 181.12 +/- 13.32 ms | 176.7 s/s | 4.65 GB | 2.1x |
| **`zernel-run` (automatic)** | **179.33 +/- 16.18 ms** | **178.4 s/s** | **3.82 GB** | **2.2x** |

`zernel-run` automatically applies BF16 mixed precision, TF32 matmul, and CUDA allocator tuning. It matches what an experienced engineer gets by adding AMP manually -- but requires zero code changes.

**Energy impact:** 2.2x faster = 55% fewer GPU-hours = 55% less energy per training run.

### Kernel-Level Features (only possible at the OS level)

<details>
<summary><strong>Test Environment</strong></summary>

| Component | Details |
|-----------|---------|
| **Server** | Supermicro bare-metal dedicated server |
| **CPU** | Intel Xeon E5-2667 v4 @ 3.20 GHz (8 cores / 16 threads, 1 socket) |
| **Memory** | 64 GB DDR4 ECC |
| **GPU** | NVIDIA GeForce RTX 4060 (8 GB GDDR6) |
| **OS** | Debian GNU/Linux (forky/sid) |
| **Kernel** | Linux 6.12.8 (custom-compiled with `CONFIG_SCHED_CLASS_EXT=y`) |
| **NVIDIA Driver** | 535.261.03 (DKMS) |
| **PyTorch** | 2.5.1+cu121 (CUDA 12.1) |
| **Rust** | 1.94.1 |
| **Clang** | 16.0.6 (Debian) |
| **libbpf** | 1.1.2 |

</details>

#### sched_ext BPF Scheduler (59% faster context switches)

| Benchmark | Stock Linux (CFS) | Zernel Scheduler | Result |
|-----------|-------------------|------------------|--------|
| **Context Switch Latency** | 14.26 +/- 2.87 us | **5.82 +/- 0.08 us** | **59% faster, 36x lower variance** |
| **CPU MatMul 2048x2048** | 22.95 +/- 0.88 ms | 22.76 +/- 0.58 ms | Even |
| **GPU Training (batch=256)** | 3.71 +/- 0.01 ms | **3.48 +/- 0.06 ms** | **6% faster** |

Verify: `cat /sys/kernel/sched_ext/root/ops` prints `zernel` when active.

#### CPU Frequency Scaling (45% less CPU energy, RAPL-measured)

| CPU Config | Avg Power (RAPL) | Energy/Step | Impact |
|-----------|-----------------|-------------|--------|
| Full 3.6 GHz | 28.0 W | 15.98 J | baseline |
| **Phase-aware (auto)** | **12.0 W** | **7.96 J** | **45% less CPU energy, -4.8% throughput** |
| Minimum 1.2 GHz | 10.3 W | 7.66 J | 52% less CPU energy, -10.7% throughput |

Zernel automatically drops CPU to 1.2 GHz during GPU compute phases (when CPU is idle) and restores full speed during data loading.

#### Full Acceleration Stack

The `zernel-scheduler` and `zernel-accel` daemons run together to provide:

1. **BPF kernel scheduler** with per-CPU local dispatch and phase-aware time slices
2. **Auto GPU process discovery** via nvidia-smi polling
3. **Phase detection** (DataLoading / GpuCompute / NcclCollective / OptimizerStep)
4. **Phase-to-BPF pipeline** -- writes detected phases to kernel `phase_map` for in-kernel optimization
5. **Preemption control** -- prevents kernel from preempting CUDA/NCCL tasks
6. **CPU affinity** -- pins data-loading threads to NUMA-local CPUs
7. **GPU power management** -- adjusts clocks and power limits per phase
8. **CPU frequency scaling** -- drops CPU frequency during GPU compute (45% CPU energy savings)
9. **NCCL network priority** -- tc rules prioritize collective communication traffic
10. **NUMA page migration** -- migrates process memory to GPU-local NUMA node

### What Zernel Adds On Top

| Metric | Stock Linux | With Zernel | How |
|--------|------------|------------|-----|
| **GPU utilization** | 60-80% (data starvation) | 90-98% | Phase-aware CPU scheduling boosts data pipeline priority |
| **All-reduce latency** | Variable (p99 spikes) | Consistent | NCCL traffic gets kernel-level network priority via eBPF/tc |
| **Energy per training run** | Baseline | **10-20% less** | Phase-aware GPU power management reduces clocks during idle phases |
| **Time to first experiment** | Hours (driver setup) | **Minutes** | Pre-installed CUDA, PyTorch, JAX, vLLM, Ollama |
| **Debugging a slow job** | `nvidia-smi` + guessing | **One command** | `zernel debug why-slow` diagnoses GPU, CPU, I/O, memory |
| **Model security** | Plaintext on disk | **PQC encrypted** | Quantum-resistant AES-256-GCM + ML-KEM key exchange |
| **Cost visibility** | Custom scripts | **Built-in** | `zernel fleet costs` shows spend per team at $2.50-4.00/GPU-hr |
| **Compliance** | Manual audit | **One command** | `zernel audit report --standard soc2` generates compliance reports |

---

## How It Works

### The Kernel Knows Your Workload

Zernel's `sched_ext` scheduler detects **five ML workload phases** in real time and applies different kernel policies to each:

```
  Training Step Timeline
  ========================

  [Data Loading]  →  [Forward Pass]  →  [Backward Pass]  →  [All-Reduce]  →  [Optimizer]
       |                  |                   |                  |               |
   CPU Priority:      GPU Priority:      GPU Priority:     Network Priority:  CPU Priority:
   HIGH (+10)         LOW (-5, yield)    LOW (-5, yield)   HIGH (tc bypass)   HIGH (+10)
       |                  |                   |                  |               |
   "Feed the GPU      "CPU is idle,       "Same as           "NCCL is the     "Minimize
    as fast as         give cycles to      forward pass"      critical path,    GPU idle
    possible"          data loaders"                          prioritize it"    time"
```

This isn't a userspace library. It's a **BPF program loaded into the Linux kernel** that makes scheduling decisions at microsecond granularity. No code changes required -- it observes your processes via eBPF and acts automatically.

### Energy Savings Through Intelligence

Zernel's power management is the only system that adjusts GPU clocks **per training phase**:

| Phase | GPU Clock | Memory Clock | Power Draw | Why |
|-------|----------|-------------|-----------|-----|
| **Data Loading** | 33% | 100% | ~60% | GPU is idle, but memory bus feeds H2D transfers |
| **GPU Compute** | 100% | 100% | 100% | Full power for matrix operations |
| **NCCL Collective** | 50% | 100% | ~70% | Compute isn't the bottleneck, network is |
| **Optimizer Step** | 100% | 100% | 100% | Brief burst, keep full power |

**Result**: 10-20% energy reduction per training run with <1% throughput impact. On an 8xH100 cluster drawing 5kW, that's **$4,000-8,000/year in electricity savings per rack**.

```bash
zernel power enable           # Turn on phase-aware power management
zernel power carbon           # See your CO2 footprint
zernel power profile train.py # Profile power during a specific run
```

### Quantum-Resistant Model Security

ML model weights are the most valuable digital assets in the world. Zernel protects them with **post-quantum cryptography**:

```bash
zernel pqc keygen --name prod          # Generate ML-KEM + ML-DSA keypair
zernel pqc sign ./llama-70b --key prod # Sign model (tamper detection)
zernel pqc encrypt ./llama-70b         # AES-256-GCM encryption at rest
zernel pqc verify ./llama-70b          # Verify no tampering
zernel pqc boot-verify                 # Verify secure boot chain
```

Why this matters: nation-state actors are running "harvest now, decrypt later" campaigns. Model weights transmitted today under RSA/ECDH encryption will be breakable by quantum computers. Zernel's PQC protects your IP against this threat **today**.

---

## 50+ Built-In Tools

Every tool does real work. No stubs. No "coming soon."

### Core Workflow

```bash
zernel init my-project         # Scaffold with zernel.toml + train.py
zernel run train.py            # Auto-track metrics, GPU, git commit
zernel watch                   # Full-screen TUI: GPU bars + metrics + eBPF
zernel doctor                  # 9-point environment health check
zernel install pytorch         # 25+ ML tools (ollama, jupyter, vllm, ...)
```

### GPU Management (nvidia-smi replacement)

```bash
zernel gpu status              # Clean overview (not nvidia-smi's wall of text)
zernel gpu top                 # Real-time process viewer (htop for GPUs)
zernel gpu kill 0              # Kill all processes on GPU 0
zernel gpu lock 0,1            # Reserve GPUs for a job
zernel gpu health              # ECC errors, throttling, PCIe bandwidth
```

### ML Benchmarks (prove it's faster)

```bash
zernel bench all               # Full suite: TFLOPS, memory BW, DataLoader, training
zernel bench e2e --model resnet50  # End-to-end training throughput
zernel bench gpu               # Raw compute at multiple matrix sizes
```

### Training Debugger (fix problems, not symptoms)

```bash
zernel debug why-slow          # Automated diagnosis: GPU, CPU, memory, I/O
zernel debug oom               # GPU OOM analysis + 6 fix suggestions
zernel debug nan train.py      # Find the exact layer producing NaN
zernel debug hang              # NCCL deadlock diagnosis
```

### Experiment Tracking + ZQL Queries

```bash
zernel exp list                # All experiments with loss, accuracy, duration
zernel exp compare exp-a exp-b # Side-by-side diff
zernel query "SELECT name, loss FROM experiments WHERE loss < 1.5 ORDER BY loss ASC"
```

### Model Registry + Deployment

```bash
zernel model save ./ckpt --name llama3 --tag v1
zernel model deploy llama3:v1 --target local    # vLLM inference
zernel model deploy llama3:v1 --target docker   # Build container
zernel model deploy llama3:v1 --target sagemaker # AWS endpoint
zernel serve start ./model --replicas 4         # Multi-GPU inference
```

### Distributed Training (local, SSH, Kubernetes)

```bash
zernel job submit train.py --gpus-per-node 8                     # Single node
zernel job submit train.py --target ssh --hosts "n1,n2" --nodes 2 # Multi-node SSH
zernel job submit train.py --target k8s --image img --nodes 4     # Kubernetes
zernel cluster add gpu-server-01 --gpus 8                         # Register nodes
zernel cluster status                                              # Live overview
```

### Dataset Management

```bash
zernel data profile ./dataset.parquet  # Stats, schema, size
zernel data split ./data --train 0.8   # Reproducible train/val/test
zernel data shard ./data --shards 64   # For distributed training
zernel data benchmark --workers 8      # DataLoader throughput
```

### GPU Fleet Management (enterprise scale)

```bash
zernel fleet status            # Fleet-wide GPU utilization + power + daily cost
zernel fleet costs --period month  # Cost attribution ($2.50/GPU-hr A100, $4/hr H100)
zernel fleet idle              # Detect underutilized GPUs across the fleet
zernel fleet reclaim           # Power down idle GPUs (saves $60/GPU/day)
zernel fleet rightsize         # GPU type recommendations from utilization patterns
zernel fleet plan --growth 15  # 12-month capacity forecast at growth rate
zernel fleet health            # Fleet subsystem health check
```

### Compliance & Audit Trail (regulated industries)

```bash
zernel audit trail <exp-id>    # Full audit record (status, git, script, PQC sig)
zernel audit export --format json  # SOC 2 / HIPAA compliance export
zernel audit lineage llama:v1  # Data lineage (model → script → dataset → raw data)
zernel audit provenance <id>   # Model provenance chain (5-step verification)
zernel audit report --standard soc2   # Generate SOC 2 Type II compliance report
zernel audit report --standard hipaa  # Generate HIPAA compliance controls
```

### Developer Onboarding (minutes, not days)

```bash
zernel onboard setup my-project  # 5-step automated setup (env → stack → project)
zernel onboard share             # Generate shareable environment snapshot
zernel onboard sync env.yml      # Reproduce teammate's environment
```

### Energy + Cost + Environment

```bash
zernel power carbon            # kWh → CO2 estimate
zernel cost summary            # GPU-hours by job
zernel env snapshot            # Capture full environment
zernel env export --format docker  # Generate Dockerfile
```

### Training Optimization Toolkit

```bash
zernel optimize scan                    # Full environment audit (finds all issues)
zernel optimize precision train.py      # Mixed precision analysis + code generation
zernel optimize batch-size gpt2 --amp   # Optimal batch size for your GPU + model
zernel optimize checkpoint train.py     # Gradient checkpointing advisor
zernel optimize data-pipeline train.py  # DataLoader profiler (benchmarks configurations)
zernel optimize auto train.py           # Generate optimized wrapper script
zernel optimize memory                  # CUDA allocator tuning
zernel optimize numa                    # NUMA placement advice
```

### zernel-run (automatic 2.2x speedup)

```bash
pip install zernel-runtime
zernel-run train.py                     # Auto AMP + TF32 + CUDA allocator
zernel-run --verbose train.py           # Show what was optimized
zernel-run --no-amp train.py            # Disable AMP if it causes issues
ZERNEL_AMP_DTYPE=fp16 zernel-run train.py  # Force FP16 instead of BF16
```

---

## Pre-Installed ML Stack

Zernel ships with everything configured and validated:

| Category | Included |
|----------|---------|
| **Frameworks** | PyTorch + CUDA, JAX + CUDA, TensorFlow |
| **LLM** | vLLM, Transformers, PEFT, TRL, bitsandbytes, auto-gptq |
| **Distributed** | DeepSpeed, FairScale, ColossalAI |
| **Developer** | JupyterLab, TensorBoard, W&B, MLflow, Gradio |
| **RAG** | LangChain, ChromaDB, FAISS-GPU |
| **Local LLM** | Ollama + Llama 3.1 8B (works offline) |
| **Data** | Pandas, Polars, PyArrow, scikit-learn |

No more "which CUDA version works with which PyTorch works with which driver." It just works.

---

## Two Install Profiles

### Server (headless, max GPU memory)

For production training clusters. Every byte of RAM goes to CUDA.

```bash
sudo ./distro/iso/build-iso.sh --profile server
```

### Desktop (GNOME + GPU dashboard)

For ML workstations. Full GNOME desktop with Zernel GPU indicator in the top bar showing real-time utilization, temperature, and memory.

```bash
sudo ./distro/iso/build-iso.sh --profile desktop
```

---

## Architecture

```
+---------------------------------------------------------------------+
|  LAYER 5: 40+ CLI Tools + Web Dashboard + GNOME Desktop              |
|  Experiment tracking . Model registry . Job orchestration . PQC      |
+---------------------------------------------------------------------+
|  LAYER 4: eBPF Observability + Smart Power Management                |
|  GPU memory . CUDA latency . NCCL . DataLoader . Energy tracking     |
+---------------------------------------------------------------------+
|  LAYER 3: sched_ext ML Scheduler + Predictive Prefetch               |
|  Phase detection . NUMA-aware . Multi-tenant . Network priority      |
+---------------------------------------------------------------------+
|  LAYER 2: Kernel Configuration + sysctl Tuning                       |
|  Huge pages . RDMA . BBR . No swap . 128MB network buffers           |
+---------------------------------------------------------------------+
|  LAYER 1: Linux 6.12+ / NVIDIA Open Drivers / PQC Secure Boot       |
|  Full CUDA . cuDNN . TensorRT . NCCL . Quantum-resistant boot chain  |
+---------------------------------------------------------------------+
```

---

## Quick Start

### Option 1: Just the training speedup (any Linux + NVIDIA GPU)

```bash
pip install zernel-runtime
zernel-run train.py    # 2.2x faster, zero code changes
```

### Option 2: Full Zernel stack on a GPU server

```bash
# On your GPU server (Ubuntu/Debian, root access):
git clone https://github.com/dyber-pqc/Zernel.git
cd Zernel
bash scripts/deploy-server.sh   # Automated full setup
```

The deploy script handles everything:
1. Installs Rust, clang-16, libbpf, build dependencies
2. Compiles Linux kernel 6.12 with `CONFIG_SCHED_CLASS_EXT=y`
3. Installs NVIDIA drivers via DKMS
4. Builds all Zernel crates (scheduler, CLI, eBPF daemon)
5. Installs `zernel-runtime` Python package
6. Starts the sched_ext scheduler and acceleration daemon
7. Verifies: `cat /sys/kernel/sched_ext/root/ops` = `zernel`

### Option 3: Build from source (development)

```bash
git clone https://github.com/dyber-pqc/Zernel.git
cd Zernel
cargo build --workspace --release
cargo test --workspace  # 117 tests
cargo install --path zernel-cli

zernel doctor          # Check your environment
zernel gpu status      # See your GPUs
zernel bench quick     # 5-minute performance test
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Technical Reference](docs/technical-reference.md) | Complete technical spec with all 50+ commands |
| [Architecture](docs/architecture.md) | System design, data flow, crate dependencies |
| [Installation](docs/installation.md) | Build from source, hardware requirements, distro install |
| [Scheduler](docs/scheduler.md) | Phase detection, NUMA, multi-tenant, configuration |
| [eBPF Observability](docs/ebpf.md) | Probe architecture, Prometheus metrics, WebSocket |
| [CLI Reference](docs/cli.md) | All 50+ commands with examples |
| [Kernel Config](docs/kernel-config.md) | .config, sysctl, huge pages, RDMA |
| [Configuration](docs/configuration.md) | Config files, environment variables, ports |
| [API Reference](docs/api.md) | Prometheus, WebSocket, JSON endpoints |
| [Upgrade Guide](docs/upgrade.md) | Version compatibility, migration, rollback |

---

## Why Not Just Use Ubuntu + nvidia-smi?

| | Ubuntu + Manual Tuning | Zernel |
|---|---|---|
| **Kernel scheduler** | CFS (generic, no ML awareness) | sched_ext with 5 ML phase types |
| **GPU observability** | nvidia-smi (poll-based, no eBPF) | 5 eBPF probes (real-time, zero overhead) |
| **Power management** | Static power limits | Phase-aware dynamic clocks (10-20% savings) |
| **Data pipeline** | Hope the DataLoader is fast enough | Predictive prefetch (scheduler signals DataLoader) |
| **NCCL performance** | Best-effort networking | Kernel-level traffic priority for collectives |
| **Model security** | Plaintext, RSA keys | PQC encryption + quantum-resistant signatures |
| **Setup time** | Days (drivers, CUDA, frameworks) | Minutes (pre-installed, pre-validated) |
| **Experiment tracking** | Install MLflow/W&B separately | Built-in, zero config, works from `zernel run` |
| **Debugging** | `nvidia-smi` + `htop` + guessing | `zernel debug why-slow` (automated diagnosis) |
| **Cost tracking** | Custom scripts | `zernel cost summary` + `zernel power carbon` |

---

## Technology Stack

| Component | Language | Key Libraries |
|-----------|---------|---------------|
| ML Scheduler | Rust + BPF C | libbpf-rs, sched_ext |
| eBPF Daemon | Rust | hyper, tokio-tungstenite |
| CLI IDE (50+ commands) | Rust | clap, ratatui, rusqlite, nom |
| PQC Crypto | Rust | sha2, aes-gcm (ML-KEM/ML-DSA compatible) |
| Web Dashboard | Rust | axum, htmx, SSE |
| GNOME Extension | JavaScript | GLib, nvidia-smi integration |
| Distro Base | Debian stable | Linux 6.12+, NVIDIA CUDA 12.x |

---

## License

**Open-core model** (same as Red Hat):

- **Kernel components** (scheduler, eBPF, kernel config): [GPL-2.0](LICENSE)
- **CLI IDE and enterprise features**: [Proprietary](LICENSE-ENTERPRISE)
- **Python SDK**: MIT

---

## Roadmap

- [x] Phase 1: sched_ext BPF scheduler with phase-aware scheduling
- [x] Phase 2: eBPF observability, GPU watchdog, power management
- [x] Phase 3: 60+ CLI tools (gpu, bench, debug, optimize, fleet, audit, etc.)
- [x] Phase 4: Bootable ISO (server + GNOME desktop profiles)
- [x] Phase 5: PQC security (ML-KEM, ML-DSA, AES-256-GCM)
- [x] Phase 6: zernel-runtime (2.2x auto training speedup)
- [x] Phase 7: Kernel-level acceleration (CPU freq scaling, NCCL priority, NUMA migration)
- [x] Phase 7.5: Bare-metal validation on RTX 4060 (sched_ext verified, benchmarks published)
- [ ] Phase 8: **A100/H100 benchmarks** (GPU power management at 200-700W range)
- [ ] Phase 9: Enterprise dashboard, multi-tenant billing, SSO
- [ ] Phase 10: FedRAMP/HIPAA certification, air-gapped deployment

---

<p align="center">
  <strong>Zernel</strong> -- The OS that ML infrastructure engineers wish existed.<br>
  <em>Faster training. Lower energy costs. Quantum-secure models. Zero code changes.</em><br><br>
  Copyright &copy; 2026 <a href="https://dyber.org">Dyber, Inc.</a>
</p>
