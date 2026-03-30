// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel debug — ML training debugger

use anyhow::{Context, Result};
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum DebugCommands {
    /// Analyze why training is slow
    WhySlow,
    /// Trace GPU out-of-memory errors
    Oom,
    /// Detect NaN gradients and trace to the source
    Nan {
        /// Training script to run with NaN detection
        script: String,
    },
    /// Detect NCCL deadlocks / straggler ranks
    Hang,
    /// Verify a checkpoint file
    Checkpoint {
        /// Path to checkpoint file
        path: String,
    },
    /// Run a script with enhanced tracing
    Trace {
        /// Script to trace
        script: String,
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
}

fn run_python(code: &str) -> Result<String> {
    let output = Command::new("python3")
        .args(["-c", code])
        .output()
        .with_context(|| "python3 not found")?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr))
}

pub async fn run(cmd: DebugCommands) -> Result<()> {
    match cmd {
        DebugCommands::WhySlow => {
            println!("Zernel Performance Diagnosis");
            println!("{}", "=".repeat(60));
            println!();

            // Check GPU utilization
            println!("[1/4] GPU Utilization...");
            if let Ok(output) = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                let data = String::from_utf8_lossy(&output.stdout);
                for line in data.lines() {
                    let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 4 {
                        let util: u32 = f[1].parse().unwrap_or(0);
                        let diagnosis = if util < 30 {
                            "LOW — likely data loading bottleneck"
                        } else if util < 70 {
                            "MEDIUM — possible data pipeline or CPU bottleneck"
                        } else {
                            "GOOD — GPU well utilized"
                        };
                        println!("  GPU {}: {}% — {}", f[0], util, diagnosis);
                    }
                }
            } else {
                println!("  nvidia-smi not available");
            }
            println!();

            // Check CPU
            println!("[2/4] CPU Utilization...");
            #[cfg(target_os = "linux")]
            {
                if let Ok(content) = std::fs::read_to_string("/proc/loadavg") {
                    let parts: Vec<&str> = content.split_whitespace().collect();
                    if let Some(load) = parts.first() {
                        let cores = std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(1);
                        let load_val: f64 = load.parse().unwrap_or(0.0);
                        let pct = (load_val / cores as f64 * 100.0) as u32;
                        let diagnosis = if pct > 90 {
                            "HIGH — CPU may be bottleneck (data preprocessing?)"
                        } else {
                            "OK"
                        };
                        println!("  Load: {load} ({cores} cores, {pct}% utilized) — {diagnosis}");
                    }
                }
            }
            #[cfg(not(target_os = "linux"))]
            println!("  (CPU check requires Linux)");
            println!();

            // Check memory
            println!("[3/4] System Memory...");
            #[cfg(target_os = "linux")]
            {
                if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                    let mut total = 0u64;
                    let mut available = 0u64;
                    for line in content.lines() {
                        if line.starts_with("MemTotal:") {
                            total = line
                                .split_whitespace()
                                .nth(1)
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0);
                        }
                        if line.starts_with("MemAvailable:") {
                            available = line
                                .split_whitespace()
                                .nth(1)
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0);
                        }
                    }
                    let used_pct = if total > 0 {
                        ((total - available) * 100 / total) as u32
                    } else {
                        0
                    };
                    let diagnosis = if used_pct > 90 {
                        "HIGH — may cause swap/OOM"
                    } else {
                        "OK"
                    };
                    println!(
                        "  {used_pct}% used ({} / {} GB) — {diagnosis}",
                        (total - available) / 1048576,
                        total / 1048576
                    );
                }
            }
            #[cfg(not(target_os = "linux"))]
            println!("  (memory check requires Linux)");
            println!();

            // Check I/O
            println!("[4/4] Recommendations");
            println!("  - If GPU util < 50%: increase DataLoader num_workers, use pin_memory=True");
            println!(
                "  - If GPU memory near limit: reduce batch size or use gradient checkpointing"
            );
            println!("  - If CPU is bottleneck: move preprocessing to GPU or use faster storage");
            println!("  - Run: zernel bench dataloader  — to measure data pipeline throughput");
            println!("  - Run: zernel gpu top  — to monitor GPU usage in real-time");
        }

        DebugCommands::Oom => {
            println!("GPU OOM Debugger");
            println!("{}", "=".repeat(60));
            println!();

            let out = run_python(
                "import torch; \
                 for i in range(torch.cuda.device_count()): \
                     total=torch.cuda.get_device_properties(i).total_mem/(1024**3); \
                     reserved=torch.cuda.memory_reserved(i)/(1024**3); \
                     allocated=torch.cuda.memory_allocated(i)/(1024**3); \
                     free=total-reserved; \
                     print(f'GPU {i}: {allocated:.1f}/{total:.1f} GB allocated, {free:.1f} GB free')"
            )?;
            println!("{out}");

            println!("Tips to fix OOM:");
            println!("  1. Reduce batch_size");
            println!("  2. Use torch.cuda.amp (mixed precision) — halves memory");
            println!("  3. Use gradient_checkpointing_enable() — trades compute for memory");
            println!("  4. Use DeepSpeed ZeRO Stage 2/3 — shards optimizer states");
            println!("  5. Use model.to(dtype=torch.bfloat16)");
            println!("  6. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True");
        }

        DebugCommands::Nan { script } => {
            println!("NaN Detector — running {script} with anomaly detection...");
            let code = format!(
                "import torch; torch.autograd.set_detect_anomaly(True); exec(open('{script}').read())"
            );
            let status = Command::new("python3").args(["-c", &code]).status()?;
            if !status.success() {
                println!("Script exited with error — check output above for NaN source.");
            }
        }

        DebugCommands::Hang => {
            println!("NCCL Hang Detector");
            println!();
            println!("Set these environment variables before training:");
            println!("  export NCCL_DEBUG=INFO");
            println!("  export NCCL_DEBUG_SUBSYS=ALL");
            println!("  export TORCH_DISTRIBUTED_DEBUG=DETAIL");
            println!("  export NCCL_ASYNC_ERROR_HANDLING=1");
            println!("  export NCCL_TIMEOUT=300  # 5 min timeout");
            println!();
            println!("Then run: zernel run train.py");
            println!("If it hangs, check which rank is stuck in the NCCL logs.");
        }

        DebugCommands::Checkpoint { path } => {
            println!("Verifying checkpoint: {path}");
            let out = run_python(&format!(
                "import torch, os; \
                 ckpt=torch.load('{path}', map_location='cpu', weights_only=False); \
                 if isinstance(ckpt, dict): \
                     print(f'Type: dict with {{len(ckpt)}} keys'); \
                     for k in list(ckpt.keys())[:20]: \
                         v=ckpt[k]; \
                         if hasattr(v,'shape'): print(f'  {{k}}: {{v.dtype}} {{list(v.shape)}}'); \
                         else: print(f'  {{k}}: {{type(v).__name__}}'); \
                     if len(ckpt)>20: print(f'  ... and {{len(ckpt)-20}} more keys'); \
                 else: print(f'Type: {{type(ckpt).__name__}}'); \
                 size=os.path.getsize('{path}')/(1024**3); \
                 print(f'Size: {{size:.2f}} GB')"
            ))?;
            println!("{out}");
        }

        DebugCommands::Trace { script, args } => {
            println!("Running {script} with enhanced tracing...");
            let mut cmd = Command::new("python3");
            cmd.args(["-u", &script]);
            cmd.args(&args);
            cmd.env("CUDA_LAUNCH_BLOCKING", "1");
            cmd.env("TORCH_SHOW_CPP_STACKTRACES", "1");
            let status = cmd.status()?;
            if !status.success() {
                println!("Script exited with code {}", status.code().unwrap_or(-1));
            }
        }
    }
    Ok(())
}
