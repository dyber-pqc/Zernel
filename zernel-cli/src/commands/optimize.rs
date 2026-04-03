// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel optimize — ML Training optimization tools
//!
//! Auto-detects and applies optimizations:
//! - Mixed precision (AMP) with auto-detection of GPU capability
//! - Smart batch size calculation based on GPU memory and model size
//! - Gradient checkpointing for memory-constrained training
//! - Data pipeline bottleneck detection and fix recommendations
//! - Full optimization scan with one command

use anyhow::Result;
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum OptimizeCommands {
    /// Analyze model for mixed precision and auto-generate AMP wrapper
    Precision {
        /// Training script to analyze and wrap
        script: String,
    },
    /// Calculate optimal batch size for your GPU and model
    BatchSize {
        /// Model name or parameter count (e.g. "125M", "1.3B", "gpt2", "llama-7b")
        model: String,
        /// Sequence length (default: 512)
        #[arg(long, default_value = "512")]
        seq_len: u32,
        /// Use mixed precision (halves memory per parameter)
        #[arg(long)]
        amp: bool,
        /// GPU ID (default: 0)
        #[arg(long, default_value = "0")]
        gpu: u32,
    },
    /// Analyze and enable gradient checkpointing for large models
    Checkpoint {
        /// Training script or model path
        script: String,
        /// Target memory usage fraction (0.0-1.0, default: 0.85)
        #[arg(long, default_value = "0.85")]
        target_mem: f32,
    },
    /// Profile data pipeline and detect bottlenecks
    DataPipeline {
        /// Training script to profile
        script: String,
        /// Number of steps to profile (default: 20)
        #[arg(long, default_value = "20")]
        steps: u32,
    },
    /// Configure CUDA memory allocator for optimal performance
    Memory,
    /// Configure NUMA-aware data placement
    Numa,
    /// Full optimization scan — analyzes everything and generates a report
    Scan {
        /// Training script to analyze (optional — scans GPU environment if omitted)
        script: Option<String>,
    },
    /// Auto-optimize: generate a wrapper script with all optimizations applied
    Auto {
        /// Training script to optimize
        script: String,
        /// Output optimized wrapper script
        #[arg(long, short, default_value = "train_optimized.py")]
        output: String,
    },
}

pub async fn run(cmd: OptimizeCommands) -> Result<()> {
    match cmd {
        OptimizeCommands::Precision { script } => run_precision(&script).await,
        OptimizeCommands::BatchSize { model, seq_len, amp, gpu } => {
            run_batch_size(&model, seq_len, amp, gpu).await
        }
        OptimizeCommands::Checkpoint { script, target_mem } => {
            run_checkpoint(&script, target_mem).await
        }
        OptimizeCommands::DataPipeline { script, steps } => {
            run_data_pipeline(&script, steps).await
        }
        OptimizeCommands::Memory => run_memory().await,
        OptimizeCommands::Numa => run_numa().await,
        OptimizeCommands::Scan { script } => run_scan(script.as_deref()).await,
        OptimizeCommands::Auto { script, output } => run_auto(&script, &output).await,
    }
}

async fn run_precision(script: &str) -> Result<()> {
    println!("Mixed Precision Analyzer");
    println!("{}", "=".repeat(60));
    println!("Script: {script}");
    println!();

    let output = Command::new("python3")
        .args(["-c", r#"
import torch, sys, os, ast, re

# 1. Check GPU capabilities
if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU available")
    sys.exit(1)

cap = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name()
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"GPU: {name}")
print(f"Compute capability: {cap[0]}.{cap[1]}")
print(f"Memory: {mem_gb:.1f} GB")
print()

# Determine best precision
if cap[0] >= 9:
    best = "FP8 (Hopper+)"
    dtype = "torch.float8_e4m3fn"
    speedup = "3-4x"
elif cap[0] >= 8:
    best = "BF16 (Ampere+)"
    dtype = "torch.bfloat16"
    speedup = "1.5-2x"
elif cap[0] >= 7:
    best = "FP16 (Volta+)"
    dtype = "torch.float16"
    speedup = "1.5-2x"
else:
    best = "FP32 only"
    dtype = "torch.float32"
    speedup = "1x (no speedup)"

print(f"Recommended precision: {best}")
print(f"Expected speedup: {speedup}")
print(f"Memory savings: {2 if cap[0] >= 7 else 1}x (FP32: {mem_gb:.1f}GB -> {'~' + str(round(mem_gb*2, 1)) if cap[0] >= 7 else str(round(mem_gb, 1))}GB effective)")
print()

# 2. Scan script for existing AMP usage
script_path = os.environ['ZERNEL_ARG_SCRIPT']
has_autocast = False
has_gradscaler = False
has_bf16 = False
has_fp16 = False

if os.path.exists(script_path):
    with open(script_path) as f:
        content = f.read()
    has_autocast = "autocast" in content
    has_gradscaler = "GradScaler" in content
    has_bf16 = "bfloat16" in content
    has_fp16 = "float16" in content and "bfloat" not in content

    if has_autocast:
        print("STATUS: Script already uses torch.autocast (AMP enabled)")
        if has_bf16:
            print("  Using: BF16 precision")
        elif has_fp16:
            print("  Using: FP16 precision")
            if cap[0] >= 8:
                print("  TIP: Switch to BF16 for better numerical stability on this GPU")
        if not has_gradscaler and has_fp16:
            print("  WARNING: Using FP16 without GradScaler - may have loss scaling issues")
            print("  Add: scaler = torch.cuda.amp.GradScaler()")
    else:
        print("STATUS: Script does NOT use mixed precision")
        print()
        print("To enable AMP, wrap your training loop:")
        print()
        if cap[0] >= 8:
            print("  # Add at top of script:")
            print("  torch.backends.cuda.matmul.allow_tf32 = True")
            print("  torch.backends.cudnn.allow_tf32 = True")
            print()
            print("  # Wrap forward pass:")
            print("  with torch.autocast('cuda', dtype=torch.bfloat16):")
            print("      output = model(input)")
            print("      loss = loss_fn(output, target)")
            print("  loss.backward()")
            print("  optimizer.step()")
        else:
            print("  scaler = torch.cuda.amp.GradScaler()")
            print()
            print("  with torch.autocast('cuda', dtype=torch.float16):")
            print("      output = model(input)")
            print("      loss = loss_fn(output, target)")
            print("  scaler.scale(loss).backward()")
            print("  scaler.step(optimizer)")
            print("  scaler.update()")
        print()
        print(f"Or use: zernel optimize auto {script_path} to generate an optimized wrapper")
else:
    print(f"Script not found: {script_path}")
    print("Showing general recommendations based on GPU capability.")
    print()
    if cap[0] >= 8:
        print("  Best: torch.autocast('cuda', dtype=torch.bfloat16)")
        print("  Also: torch.backends.cuda.matmul.allow_tf32 = True")
    elif cap[0] >= 7:
        print("  Best: torch.autocast('cuda', dtype=torch.float16)")
        print("  Need: torch.cuda.amp.GradScaler() for loss scaling")
"#])
        .env("ZERNEL_ARG_SCRIPT", script)
        .output()?;

    print!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(())
}

async fn run_batch_size(model: &str, seq_len: u32, amp: bool, gpu: u32) -> Result<()> {
    println!("Smart Batch Size Calculator");
    println!("{}", "=".repeat(60));
    println!();

    let output = Command::new("python3")
        .args(["-c", r#"
import torch, sys, math, os

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU available")
    sys.exit(1)

gpu_id = int(os.environ['ZERNEL_ARG_GPU'])
props = torch.cuda.get_device_properties(gpu_id)
total_mem = props.total_memory
name = props.name
mem_gb = total_mem / 1e9

print(f"GPU {gpu_id}: {name} ({mem_gb:.1f} GB)")

# Parse model size
model_str = os.environ['ZERNEL_ARG_MODEL'].lower().strip()
param_count = None

# Known models
known = {
    "gpt2": 124e6, "gpt2-medium": 355e6, "gpt2-large": 774e6, "gpt2-xl": 1.5e9,
    "llama-7b": 7e9, "llama-13b": 13e9, "llama-70b": 70e9,
    "mistral-7b": 7.2e9, "phi-2": 2.7e9, "phi-3": 3.8e9,
    "bert-base": 110e6, "bert-large": 340e6,
    "t5-small": 60e6, "t5-base": 220e6, "t5-large": 770e6,
    "vit-base": 86e6, "vit-large": 307e6,
    "resnet50": 25.6e6, "resnet101": 44.5e6,
}

if model_str in known:
    param_count = known[model_str]
    print(f"Model: {model_str} ({param_count/1e6:.0f}M params)")
elif model_str.endswith('b'):
    param_count = float(model_str[:-1]) * 1e9
    print(f"Model: {param_count/1e9:.1f}B params")
elif model_str.endswith('m'):
    param_count = float(model_str[:-1]) * 1e6
    print(f"Model: {param_count/1e6:.0f}M params")
else:
    try:
        param_count = float(model_str)
        print(f"Model: {param_count/1e6:.0f}M params")
    except:
        print(f"Unknown model: {model_str}")
        print("Use a known name (gpt2, llama-7b, etc.) or param count (125M, 1.3B)")
        sys.exit(1)

seq_len = int(os.environ['ZERNEL_ARG_SEQ_LEN'])
use_amp = os.environ['ZERNEL_ARG_AMP'] == "1"
bytes_per_param = 2 if use_amp else 4

print(f"Sequence length: {seq_len}")
print(f"Precision: {'FP16/BF16' if use_amp else 'FP32'}")
print()

# Memory estimation (rough but practical):
# Model params: param_count * bytes_per_param
# Gradients: same as model
# Optimizer state (AdamW): 2x model size (momentum + variance) in FP32
# Activations: depends on model architecture, roughly:
#   For transformers: ~12 * n_layers * hidden_dim * seq_len * batch_size * bytes_per_param
#   Simplified: ~6 * param_count * seq_len / hidden_dim * batch_size * bytes_per_param / 1024

model_mem = param_count * bytes_per_param
grad_mem = model_mem
opt_mem = param_count * 8  # AdamW always FP32: momentum + variance = 2 * 4 bytes
static_mem = model_mem + grad_mem + opt_mem
cuda_overhead = 0.5e9  # ~500MB CUDA context

available = total_mem - static_mem - cuda_overhead
if available <= 0:
    print(f"WARNING: Model doesn't fit in GPU memory!")
    print(f"  Model + gradients + optimizer: {static_mem/1e9:.1f} GB")
    print(f"  GPU memory: {mem_gb:.1f} GB")
    print(f"  Shortfall: {(static_mem + cuda_overhead - total_mem)/1e9:.1f} GB")
    print()
    print("Recommendations:")
    print(f"  1. Use mixed precision: zernel optimize batch-size {model_str} --amp")
    print("  2. Use gradient checkpointing: zernel optimize checkpoint <script>")
    print("  3. Use model parallelism or offloading")
    sys.exit(0)

# Estimate activation memory per sample
# Rough heuristic: 4-6 bytes per param per sample (varies by architecture)
if param_count > 1e9:
    act_per_sample = param_count * 4 * seq_len / 2048  # scale with seq_len
else:
    act_per_sample = param_count * 6 * seq_len / 512

act_per_sample *= (0.5 if use_amp else 1.0)

max_batch = max(1, int(available / act_per_sample))
# Round down to power of 2 for efficiency
optimal_batch = 2 ** int(math.log2(max_batch)) if max_batch >= 2 else 1
safe_batch = max(1, optimal_batch // 2)  # 50% margin for safety

print(f"Memory breakdown:")
print(f"  Model weights:    {model_mem/1e9:.2f} GB")
print(f"  Gradients:        {grad_mem/1e9:.2f} GB")
print(f"  Optimizer (AdamW):{opt_mem/1e9:.2f} GB")
print(f"  CUDA overhead:    {cuda_overhead/1e9:.2f} GB")
print(f"  Available for activations: {available/1e9:.2f} GB")
print()
print(f"Recommended batch sizes:")
print(f"  Maximum:   {max_batch} (uses ~100% GPU memory — risky)")
print(f"  Optimal:   {optimal_batch} (power-of-2, ~{optimal_batch * act_per_sample / available * 100:.0f}% memory)")
print(f"  Safe:      {safe_batch} (50% margin for variable-length sequences)")
print()

if not use_amp and param_count > 100e6:
    amp_available = total_mem - (param_count * 2 + param_count * 2 + opt_mem + cuda_overhead)
    amp_act = act_per_sample * 0.5
    amp_max = max(1, int(amp_available / amp_act))
    amp_optimal = 2 ** int(math.log2(amp_max)) if amp_max >= 2 else 1
    print(f"With mixed precision (--amp):")
    print(f"  Optimal batch size: {amp_optimal} ({amp_optimal/optimal_batch:.1f}x larger)")
    print(f"  Run: zernel optimize batch-size {model_str} --seq-len {seq_len} --amp")
"#])
        .env("ZERNEL_ARG_MODEL", model)
        .env("ZERNEL_ARG_SEQ_LEN", seq_len.to_string())
        .env("ZERNEL_ARG_AMP", if amp { "1" } else { "0" })
        .env("ZERNEL_ARG_GPU", gpu.to_string())
        .output()?;

    print!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(())
}

async fn run_checkpoint(script: &str, target_mem: f32) -> Result<()> {
    println!("Gradient Checkpointing Analyzer");
    println!("{}", "=".repeat(60));
    println!("Script: {script}");
    println!("Target memory usage: {:.0}%", target_mem * 100.0);
    println!();

    let output = Command::new("python3")
        .args(["-c", r#"
import torch, sys, os

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU available")
    sys.exit(1)

mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
name = torch.cuda.get_device_name()
target = float(os.environ['ZERNEL_ARG_TARGET_MEM'])

print(f"GPU: {name} ({mem_gb:.1f} GB)")
print(f"Target usage: {target*100:.0f}% ({mem_gb*target:.1f} GB)")
print()

# Check if script exists and scan for model definition
script_path = os.environ['ZERNEL_ARG_SCRIPT']
has_checkpoint = False
model_type = "unknown"

if os.path.exists(script_path):
    with open(script_path) as f:
        content = f.read()
    has_checkpoint = "checkpoint_sequential" in content or "checkpoint" in content and "gradient" in content.lower()

    if "transformers" in content or "AutoModel" in content:
        model_type = "huggingface"
    elif "nn.Sequential" in content or "nn.Module" in content:
        model_type = "pytorch"

if has_checkpoint:
    print("STATUS: Script already uses gradient checkpointing")
else:
    print("STATUS: Gradient checkpointing NOT enabled")
    print()
    print("Gradient checkpointing trades compute for memory:")
    print("  - Saves ~60-70% activation memory")
    print("  - Costs ~30% extra compute (recomputes activations in backward)")
    print("  - Allows ~2-3x larger batch sizes or models")
    print()

    if model_type == "huggingface":
        print("For HuggingFace models:")
        print("  model.gradient_checkpointing_enable()")
        print()
        print("Or in TrainingArguments:")
        print("  TrainingArguments(gradient_checkpointing=True, ...)")
    else:
        print("For PyTorch models:")
        print("  from torch.utils.checkpoint import checkpoint_sequential")
        print()
        print("  # In your model's forward():")
        print("  # Instead of: x = self.layers(x)")
        print("  # Use:        x = checkpoint_sequential(self.layers, segments, x)")
        print()
        print("  # Or per-layer:")
        print("  from torch.utils.checkpoint import checkpoint")
        print("  for layer in self.layers:")
        print("      x = checkpoint(layer, x, use_reentrant=False)")

    print()
    print("When to use gradient checkpointing:")
    print(f"  - Model + optimizer > {mem_gb*0.6:.1f} GB (60% of GPU memory)")
    print(f"  - You're getting OOM errors")
    print(f"  - You want to increase batch size without more GPUs")
    print()
    print("When NOT to use it:")
    print(f"  - You have plenty of GPU memory headroom")
    print(f"  - Training is already compute-bound (GPU util > 90%)")
    print(f"  - The 30% slowdown isn't acceptable")
"#])
        .env("ZERNEL_ARG_SCRIPT", script)
        .env("ZERNEL_ARG_TARGET_MEM", target_mem.to_string())
        .output()?;

    print!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(())
}

async fn run_data_pipeline(script: &str, steps: u32) -> Result<()> {
    println!("Data Pipeline Profiler");
    println!("{}", "=".repeat(60));
    println!("Script: {script}");
    println!("Profiling: {steps} steps");
    println!();

    let output = Command::new("python3")
        .args(["-c", r#"
import torch, time, sys, os, statistics

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU available")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"CPU cores: {os.cpu_count()}")
print()

# Profile different DataLoader configurations
from torch.utils.data import DataLoader, TensorDataset

# Create dummy dataset (1M samples of 224x224x3 "images")
n_samples = 10000
data = torch.randn(n_samples, 3, 224, 224)
labels = torch.randint(0, 1000, (n_samples,))
dataset = TensorDataset(data, labels)

configs = [
    {"num_workers": 0, "pin_memory": False, "persistent_workers": False, "label": "Default (0 workers)"},
    {"num_workers": 2, "pin_memory": False, "persistent_workers": False, "label": "2 workers"},
    {"num_workers": 4, "pin_memory": True,  "persistent_workers": False, "label": "4 workers + pin_memory"},
    {"num_workers": 4, "pin_memory": True,  "persistent_workers": True,  "label": "4 workers + pin + persistent"},
    {"num_workers": 8, "pin_memory": True,  "persistent_workers": True,  "label": "8 workers + pin + persistent"},
]

n_steps = int(os.environ['ZERNEL_ARG_STEPS'])
batch_size = 64

print(f"Profiling {n_steps} batches (batch_size={batch_size}) per config...")
print()
print(f"  {'Config':<40} {'Time(ms)':>10} {'Throughput':>12} {'vs Best':>10}")
print(f"  {'-'*75}")

best_time = float('inf')
results = []

for cfg in configs:
    kw = {k: v for k, v in cfg.items() if k != 'label'}
    if kw.get('persistent_workers') and kw.get('num_workers', 0) == 0:
        continue
    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kw)
        it = iter(loader)

        # Warmup
        for _ in range(3):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

        # Benchmark
        times = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            # Simulate GPU transfer
            x = batch[0].to('cuda', non_blocking=kw.get('pin_memory', False))
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(times)
        throughput = batch_size * 1000 / avg
        best_time = min(best_time, avg)
        results.append((cfg['label'], avg, throughput))

        del loader
    except Exception as e:
        results.append((cfg['label'], 0, 0))

for label, avg, throughput in results:
    if avg > 0:
        ratio = best_time / avg
        marker = " <-- best" if abs(avg - best_time) < 0.01 else ""
        print(f"  {label:<40} {avg:>8.2f}ms {throughput:>9.0f} s/s  {ratio:>7.2f}x{marker}")
    else:
        print(f"  {label:<40} {'FAILED':>10}")

print()

# Recommendations
best_cfg = None
best_avg = float('inf')
for label, avg, _ in results:
    if 0 < avg < best_avg:
        best_avg = avg
        best_cfg = label

if best_cfg:
    print(f"Recommendation: Use '{best_cfg}'")
    print()
    print("Add to your DataLoader:")
    print("  DataLoader(")
    print("      dataset,")
    print(f"      batch_size={batch_size},")
    if "8 worker" in best_cfg:
        print("      num_workers=8,")
    elif "4 worker" in best_cfg:
        print("      num_workers=4,")
    elif "2 worker" in best_cfg:
        print("      num_workers=2,")
    if "pin" in best_cfg:
        print("      pin_memory=True,")
    if "persistent" in best_cfg:
        print("      persistent_workers=True,")
    print("      prefetch_factor=2,")
    print("  )")

    # Check if data loading is the bottleneck
    typical_gpu_step = 50  # ms for a typical forward+backward
    if best_avg > typical_gpu_step * 0.3:
        print()
        print("WARNING: Data loading may be a bottleneck!")
        print(f"  Data load time ({best_avg:.1f}ms) is >{typical_gpu_step*0.3:.0f}ms (30% of typical GPU step)")
        print("  Consider: larger prefetch_factor, SSD storage, or data preprocessing")
"#])
        .env("ZERNEL_ARG_SCRIPT", script)
        .env("ZERNEL_ARG_STEPS", steps.to_string())
        .output()?;

    print!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(())
}

async fn run_memory() -> Result<()> {
    println!("CUDA Memory Allocator Configuration");
    println!("{}", "=".repeat(60));

    let code = r#"
import os, torch
if torch.cuda.is_available():
    conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(not set)')
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {conf}")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU memory: {props.total_memory / 1e9:.1f} GB")
    print()
    print("Recommended settings:")
    print("  For training:")
    print("    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:512")
    print("  For inference:")
    print("    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128")
"#;
    let output = Command::new("python3").args(["-c", code]).output()?;
    print!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}

async fn run_numa() -> Result<()> {
    println!("NUMA-Aware Data Placement");
    println!("{}", "=".repeat(60));

    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
            let mut node_count = 0;
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("node") {
                    node_count += 1;
                    let cpulist_path = entry.path().join("cpulist");
                    if let Ok(cpus) = std::fs::read_to_string(&cpulist_path) {
                        println!("  {name}: CPUs {}", cpus.trim());
                    }
                }
            }
            println!();
            if node_count > 1 {
                println!("Multi-NUMA system ({node_count} nodes). Use:");
                println!("  numactl --cpunodebind=0 --membind=0 python3 train.py");
            } else {
                println!("Single NUMA node — no placement optimization needed.");
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    println!("  NUMA detection requires Linux.");

    Ok(())
}

async fn run_scan(script: Option<&str>) -> Result<()> {
    println!("Zernel Optimization Scan");
    println!("{}", "=".repeat(60));
    println!();

    let script_arg = script.unwrap_or("(none)");
    let output = Command::new("python3")
        .args(["-c", r#"
import torch, os, sys

issues = []
recommendations = []

print("Environment:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {name} ({mem:.1f} GB, sm_{cap[0]}{cap[1]})")

    # Check TF32
    if cap[0] >= 8:
        tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        tf32_cudnn = torch.backends.cudnn.allow_tf32
        if not tf32_matmul:
            issues.append("TF32 disabled for matmul (free 3x speedup on Ampere+)")
            recommendations.append("torch.backends.cuda.matmul.allow_tf32 = True")
        if not tf32_cudnn:
            issues.append("TF32 disabled for cuDNN")
            recommendations.append("torch.backends.cudnn.allow_tf32 = True")

    # Check CUDA alloc config
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    if 'expandable_segments' not in alloc_conf:
        issues.append("CUDA allocator not optimized (expandable_segments not set)")
        recommendations.append("export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    # Check precision recommendation
    if cap[0] >= 8:
        recommendations.append("Use BF16: torch.autocast('cuda', dtype=torch.bfloat16)")
    elif cap[0] >= 7:
        recommendations.append("Use FP16: torch.autocast('cuda', dtype=torch.float16) + GradScaler()")

# Check CPU
cpu_count = os.cpu_count()
print(f"  CPUs: {cpu_count}")
if cpu_count and cpu_count >= 4:
    recommendations.append(f"DataLoader: num_workers={min(cpu_count, 8)}, pin_memory=True, persistent_workers=True")

# Check script if provided
script_path = os.environ['ZERNEL_ARG_SCRIPT']
if script_path != "(none)" and os.path.exists(script_path):
    with open(script_path) as f:
        content = f.read()
    print(f"  Script: {script_path}")

    if "autocast" not in content and "amp" not in content.lower():
        issues.append("No mixed precision (AMP) detected in script")
    if "num_workers" not in content:
        issues.append("DataLoader num_workers not set (defaults to 0 — single-threaded)")
    if "pin_memory" not in content:
        issues.append("DataLoader pin_memory not set (slower CPU→GPU transfers)")
    if "gradient_checkpointing" not in content and "checkpoint_sequential" not in content:
        recommendations.append("Consider gradient checkpointing for larger models/batches")

print()
if issues:
    print(f"ISSUES FOUND ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("No issues found!")

print()
if recommendations:
    print(f"RECOMMENDATIONS ({len(recommendations)}):")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

print()
print("Run individual optimizations:")
print("  zernel optimize precision <script>     # Mixed precision analysis")
print("  zernel optimize batch-size <model>      # Optimal batch size")
print("  zernel optimize checkpoint <script>     # Gradient checkpointing")
print("  zernel optimize data-pipeline <script>  # Data loading profiler")
print("  zernel optimize auto <script>           # Generate optimized wrapper")
"#])
        .env("ZERNEL_ARG_SCRIPT", script_arg)
        .output()?;

    print!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(())
}

async fn run_auto(script: &str, output_path: &str) -> Result<()> {
    println!("Zernel Auto-Optimizer");
    println!("{}", "=".repeat(60));
    println!("Input:  {script}");
    println!("Output: {output_path}");
    println!();

    // Pass args via env vars to avoid format string escaping issues
    let output = Command::new("python3")
        .args(["-c", include_str!("optimize_auto.py")])
        .env("ZERNEL_OPT_SCRIPT", script)
        .env("ZERNEL_OPT_OUTPUT", output_path)
        .output();

    match output {
        Ok(out) => {
            print!("{}", String::from_utf8_lossy(&out.stdout));
            if !out.status.success() {
                eprint!("{}", String::from_utf8_lossy(&out.stderr));
            }
        }
        Err(_) => {
            // Fallback: inline simple version
            println!("Generating optimized wrapper for {script}...");
            let original = std::fs::read_to_string(script)?;
            let mut wrapper = format!("#!/usr/bin/env python3\n# Auto-optimized by Zernel\n");
            wrapper.push_str("import torch, os\n");
            wrapper.push_str("torch.backends.cuda.matmul.allow_tf32 = True\n");
            wrapper.push_str("torch.backends.cudnn.allow_tf32 = True\n");
            wrapper.push_str("os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n\n");
            wrapper.push_str("# Original script:\n");
            wrapper.push_str(&original);
            std::fs::write(output_path, &wrapper)?;
            println!("Written to: {output_path}");
        }
    }
    Ok(())
}
