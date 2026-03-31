// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel profile — Full training pipeline profiler
//!
//! Detailed breakdown of where every millisecond goes in a training step.
//! Shows waterfall chart in the terminal.

use anyhow::{Context, Result};
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum ProfileCommands {
    /// Profile a training script (runs 10 steps with tracing)
    Run {
        /// Training script
        script: String,
        /// Number of profiling steps
        #[arg(long, default_value = "10")]
        steps: u32,
    },
    /// Show GPU utilization timeline
    Timeline {
        /// Duration in seconds
        #[arg(long, default_value = "30")]
        duration: u32,
    },
    /// Profile CUDA operations
    Cuda {
        /// Training script
        script: String,
    },
}

pub async fn run(cmd: ProfileCommands) -> Result<()> {
    match cmd {
        ProfileCommands::Run { script, steps } => {
            println!("Zernel Training Profiler");
            println!("{}", "=".repeat(60));
            println!("  Script: {script}");
            println!("  Steps:  {steps}");
            println!();

            // Generate a profiling wrapper script
            let profile_code = format!(
                r#"
import torch
import torch.autograd.profiler as profiler
import time
import sys
import os

# Enable CUDA synchronous execution for accurate timing
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("Running profiler...")
print()

# Import and run the training script with profiling
with profiler.profile(use_cuda=True, profile_memory=True, record_shapes=True) as prof:
    # Execute the training script
    exec(open('{script}').read())

# Print profiler results
print()
print("=" * 70)
print("ZERNEL TRAINING PROFILE")
print("-" * 70)
print()

# Table sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

print()
print("-" * 70)

# Memory summary
if torch.cuda.is_available():
    print()
    print("GPU Memory Summary:")
    print(f"  Peak allocated: {{torch.cuda.max_memory_allocated() / 1e9:.2f}} GB")
    print(f"  Peak reserved:  {{torch.cuda.max_memory_reserved() / 1e9:.2f}} GB")
    print(f"  Current:        {{torch.cuda.memory_allocated() / 1e9:.2f}} GB")

# Export chrome trace
trace_file = '/tmp/zernel_profile_trace.json'
prof.export_chrome_trace(trace_file)
print(f)
print(f"Chrome trace saved to: {{trace_file}}")
print("View in Chrome: chrome://tracing → Load → select the file")
"#
            );

            let wrapper_path = "/tmp/zernel_profile_wrapper.py";
            std::fs::write(wrapper_path, &profile_code)?;

            let status = tokio::process::Command::new("python3")
                .arg(wrapper_path)
                .env("CUDA_LAUNCH_BLOCKING", "1")
                .status()
                .await
                .with_context(|| format!("failed to run {script}"))?;

            if !status.success() {
                // Fallback: just run with basic timing
                println!("PyTorch profiler failed. Running basic timing analysis...");
                println!();

                let basic_code = r#"
import torch, time

if torch.cuda.is_available():
    # Measure H2D transfer
    data = torch.randn(256, 3, 224, 224)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        gpu_data = data.cuda()
        del gpu_data
    torch.cuda.synchronize()
    t1 = time.time()
    h2d_ms = (t1 - t0) / 10 * 1000
    print(f"  H2D transfer:     {{h2d_ms:.2f}} ms/batch")

    # Measure allocation
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        x = torch.empty(1024*1024*64, device='cuda')
        del x
    torch.cuda.synchronize()
    t1 = time.time()
    alloc_us = (t1 - t0) / 100 * 1e6
    print(f"  GPU alloc (256MB): {{alloc_us:.0f}} us")

    # Measure sync overhead
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        torch.cuda.synchronize()
    t1 = time.time()
    sync_us = (t1 - t0) / 1000 * 1e6
    print(f"  CUDA sync:        {{sync_us:.1f}} us")

    print()
    print(f"  Peak GPU memory:  {{torch.cuda.max_memory_allocated() / 1e9:.2f}} GB")
"#;
                let _ = tokio::process::Command::new("python3")
                    .args(["-c", basic_code])
                    .status()
                    .await;
            }
        }

        ProfileCommands::Timeline { duration } => {
            println!("GPU Utilization Timeline ({duration}s)");
            println!("{}", "=".repeat(60));
            println!();

            let start = std::time::Instant::now();
            let mut samples = Vec::new();

            while start.elapsed().as_secs() < duration as u64 {
                let output = Command::new("nvidia-smi")
                    .args([
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ])
                    .output();

                if let Ok(o) = output {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    if let Some(line) = stdout.lines().next() {
                        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                        if f.len() >= 3 {
                            let util: u32 = f[0].parse().unwrap_or(0);
                            let elapsed = start.elapsed().as_secs();
                            let bar_len = (util as usize * 40) / 100;
                            let bar =
                                format!("{}{}", "#".repeat(bar_len), " ".repeat(40 - bar_len));
                            println!(
                                "  {:>3}s [{bar}] {:>3}% {:>5}MB {:>5}W",
                                elapsed, util, f[1], f[2]
                            );
                            samples.push(util);
                        }
                    }
                }

                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }

            if !samples.is_empty() {
                let avg: u32 = samples.iter().sum::<u32>() / samples.len() as u32;
                let max = samples.iter().max().unwrap_or(&0);
                let min = samples.iter().min().unwrap_or(&0);
                println!();
                println!("Summary: avg={avg}% min={min}% max={max}%");

                if avg < 50 {
                    println!("  → GPU underutilized. Run: zernel debug why-slow");
                }
            }
        }

        ProfileCommands::Cuda { script } => {
            println!("CUDA Operation Profile: {script}");
            println!("{}", "=".repeat(60));
            println!();
            println!("Running with NVIDIA Nsight Systems...");

            let nsys_check = Command::new("nsys").arg("--version").output();
            match nsys_check {
                Ok(o) if o.status.success() => {
                    let status = tokio::process::Command::new("nsys")
                        .args([
                            "profile",
                            "--stats=true",
                            "--output=/tmp/zernel_nsys",
                            "python3",
                            &script,
                        ])
                        .status()
                        .await?;

                    if status.success() {
                        println!("Profile saved to: /tmp/zernel_nsys.nsys-rep");
                        println!("View with: nsys-ui /tmp/zernel_nsys.nsys-rep");
                    }
                }
                _ => {
                    println!("  nsys not found. Using PyTorch CUDA profiler instead...");
                    let status = tokio::process::Command::new("python3")
                        .args(["-c", &format!(
                            "import torch; torch.cuda.cudart().cudaProfilerStart(); exec(open('{script}').read()); torch.cuda.cudart().cudaProfilerStop()"
                        )])
                        .env("CUDA_LAUNCH_BLOCKING", "1")
                        .status()
                        .await?;
                    let _ = status;
                }
            }
        }
    }
    Ok(())
}
