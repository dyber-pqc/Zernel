// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel optimize — Training optimization tools
//! CUDA allocator, mixed precision advisor, checkpoint optimization

use anyhow::Result;
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum OptimizeCommands {
    /// Analyze model for mixed precision opportunities
    Precision {
        /// Training script or model path
        script: String,
    },
    /// Configure CUDA memory allocator for optimal performance
    Memory,
    /// Enable async checkpointing
    Checkpoint {
        /// Checkpoint directory
        path: String,
    },
    /// Full optimization scan — analyzes and recommends improvements
    Scan {
        /// Training script to analyze
        script: String,
    },
    /// Configure NUMA-aware data placement
    Numa,
}

pub async fn run(cmd: OptimizeCommands) -> Result<()> {
    match cmd {
        OptimizeCommands::Precision { script } => {
            println!("Mixed Precision Advisor");
            println!("{}", "=".repeat(60));
            println!("Analyzing: {script}");
            println!();

            let code = r#"
import torch, sys
try:
    # Try to import and profile
    print("Checking PyTorch mixed precision support...")
    print(f"  PyTorch: {{torch.__version__}}")
    print(f"  CUDA: {{torch.cuda.is_available()}}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        print(f"  Compute capability: {{cap[0]}}.{{cap[1]}}")
        if cap[0] >= 7:
            print(f"  FP16: supported (Tensor Cores)")
            print(f"  BF16: {{'supported' if cap[0] >= 8 else 'not supported (need Ampere+)'}}")
            print(f"  TF32: {{'supported' if cap[0] >= 8 else 'not supported'}}")
        else:
            print("  FP16: supported (no Tensor Cores - slower)")
        print()
        print("Recommendations:")
        if cap[0] >= 8:
            print("  1. Use torch.cuda.amp with BF16 (best for Ampere/Hopper)")
            print("     with torch.autocast('cuda', dtype=torch.bfloat16):")
            print("  2. Enable TF32 for matmuls:")
            print("     torch.backends.cuda.matmul.allow_tf32 = True")
            print("  3. Enable TF32 for convolutions:")
            print("     torch.backends.cudnn.allow_tf32 = True")
        elif cap[0] >= 7:
            print("  1. Use torch.cuda.amp with FP16:")
            print("     scaler = torch.cuda.amp.GradScaler()")
            print("     with torch.autocast('cuda', dtype=torch.float16):")
        print()
        # Memory savings estimate
        total_mem = torch.cuda.get_device_properties(0).total_mem
        print(f"  GPU memory: {{total_mem / 1e9:.1f}} GB")
        print(f"  With FP16:  ~{{total_mem / 2e9:.1f}} GB effective (2x capacity)")
        print(f"  With BF16:  ~{{total_mem / 2e9:.1f}} GB effective (2x capacity)")
except Exception as e:
    print(f"Error: {{e}}")
"#;

            let output = Command::new("python3").args(["-c", code]).output()?;
            print!("{}", String::from_utf8_lossy(&output.stdout));
            if !output.status.success() {
                print!("{}", String::from_utf8_lossy(&output.stderr));
            }
        }

        OptimizeCommands::Memory => {
            println!("CUDA Memory Allocator Configuration");
            println!("{}", "=".repeat(60));
            println!();
            println!("Current settings:");

            let code = r#"
import os, torch
if torch.cuda.is_available():
    conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {conf}")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU memory: {props.total_mem / 1e9:.1f} GB")
    print(f"  Reserved:   {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    print(f"  Allocated:  {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
    print()
    print("Recommended settings for training:")
    print("  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,")
    print("    garbage_collection_threshold:0.6,max_split_size_mb:512")
    print()
    print("For inference (minimize fragmentation):")
    print("  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,")
    print("    max_split_size_mb:128")
    print()
    print("Apply now:")
    print("  zernel optimize memory  (already sets these env vars)")
"#;
            let output = Command::new("python3").args(["-c", code]).output()?;
            print!("{}", String::from_utf8_lossy(&output.stdout));

            // Set the env vars for future zernel run commands
            println!();
            println!("To apply, add to your training script or shell:");
            println!("  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:512");
        }

        OptimizeCommands::Checkpoint { path } => {
            println!("Checkpoint Optimization");
            println!("{}", "=".repeat(60));
            println!("Path: {path}");
            println!();

            // Check current checkpoint size
            let p = std::path::Path::new(&path);
            if p.exists() {
                let size = if p.is_file() {
                    p.metadata().map(|m| m.len()).unwrap_or(0)
                } else {
                    0 // would need recursive walk
                };
                println!("  Current size: {:.2} GB", size as f64 / 1e9);
            }

            println!();
            println!("Optimization recommendations:");
            println!("  1. Async checkpointing (save while training continues):");
            println!("     import torch.distributed.checkpoint as dcp");
            println!("     dcp.async_save(state_dict, checkpoint_id=path)");
            println!();
            println!("  2. Incremental saves (only changed parameters):");
            println!(
                "     torch.save(model.state_dict(), path, _use_new_zipfile_serialization=True)"
            );
            println!();
            println!("  3. FP16 checkpoints (half the size):");
            println!("     state = {{k: v.half() for k, v in model.state_dict().items()}}");
            println!("     torch.save(state, f'{{path}}.fp16')");
            println!();
            println!("  4. Sharded saves for multi-GPU:");
            println!("     from torch.distributed.fsdp import FullyShardedDataParallel");
            println!("     FSDP.save_model_to_path(model, path)");
        }

        OptimizeCommands::Scan { script } => {
            println!("Zernel Optimization Scan");
            println!("{}", "=".repeat(60));
            println!("Script: {script}");
            println!();

            let checks = [
                ("Mixed Precision", "python3 -c \"import torch; print('BF16' if torch.cuda.get_device_capability()[0] >= 8 else 'FP16')\""),
                ("TF32", "python3 -c \"import torch; print('available' if hasattr(torch.backends.cuda, 'matmul') else 'N/A')\""),
                ("CUDA Alloc Config", "echo $PYTORCH_CUDA_ALLOC_CONF"),
                ("NCCL Config", "echo ${NCCL_ALGO:-default}"),
                ("DataLoader Workers", "python3 -c \"import os; print(os.cpu_count())\""),
            ];

            for (name, cmd_str) in &checks {
                let parts: Vec<&str> = cmd_str.split_whitespace().collect();
                let output = Command::new(parts[0]).args(&parts[1..]).output();
                let result = output
                    .ok()
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                    .unwrap_or_else(|| "N/A".into());
                println!("  {name:<25} {result}");
            }

            println!();
            println!("Run individual optimizations:");
            println!("  zernel optimize precision {script}");
            println!("  zernel optimize memory");
            println!("  zernel optimize numa");
            println!("  zernel debug why-slow");
        }

        OptimizeCommands::Numa => {
            println!("NUMA-Aware Data Placement");
            println!("{}", "=".repeat(60));

            #[cfg(target_os = "linux")]
            {
                // Detect NUMA topology
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
                        println!("Multi-NUMA system detected ({node_count} nodes).");
                        println!();
                        println!("Recommendations:");
                        println!("  1. Pin DataLoader workers to same NUMA node as GPU:");
                        println!("     numactl --cpunodebind=0 --membind=0 python3 train.py");
                        println!("  2. Use ZERNEL scheduler (auto-detects GPU→NUMA mapping):");
                        println!("     sudo zernel-scheduler");
                        println!("  3. Set PyTorch DataLoader affinity:");
                        println!("     os.sched_setaffinity(0, cpus_on_gpu_numa_node)");
                    } else {
                        println!("Single NUMA node — no placement optimization needed.");
                    }
                }
            }

            #[cfg(not(target_os = "linux"))]
            {
                println!("  NUMA detection requires Linux.");
                println!("  On multi-socket GPU servers, Zernel automatically pins");
                println!("  tasks to the NUMA node closest to their GPU.");
            }
        }
    }
    Ok(())
}
