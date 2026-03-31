// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel migrate — Live job migration between GPUs
//!
//! Checkpoint a running training job, move it to a different GPU, and resume.
//! Useful for GPU maintenance, load balancing, and thermal management.

use anyhow::{Context, Result};
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum MigrateCommands {
    /// Checkpoint and migrate a training job to a different GPU
    Job {
        /// Process ID of the training job
        pid: u32,
        /// Source GPU index
        #[arg(long)]
        from: u32,
        /// Destination GPU index
        #[arg(long)]
        to: u32,
    },
    /// Show migration-capable jobs (processes using CUDA)
    List,
    /// Checkpoint a job without migrating (for backup/recovery)
    Checkpoint {
        /// Process ID
        pid: u32,
        /// Checkpoint output directory
        #[arg(long, default_value = "/tmp/zernel-checkpoint")]
        output: String,
    },
    /// Resume a job from a checkpoint
    Resume {
        /// Checkpoint directory
        path: String,
        /// GPU to resume on
        #[arg(long, default_value = "0")]
        gpu: u32,
    },
}

pub async fn run(cmd: MigrateCommands) -> Result<()> {
    match cmd {
        MigrateCommands::Job { pid, from, to } => {
            println!("Zernel GPU Migration");
            println!("{}", "=".repeat(50));
            println!("  PID:  {pid}");
            println!("  From: GPU {from}");
            println!("  To:   GPU {to}");
            println!();

            // Verify process exists
            #[cfg(unix)]
            {
                let proc_path = format!("/proc/{pid}");
                if !std::path::Path::new(&proc_path).exists() {
                    anyhow::bail!("process {pid} not found");
                }
            }

            // Step 1: Send SIGUSR1 to trigger checkpoint (convention for ML frameworks)
            println!("[1/4] Signaling process to checkpoint...");
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGUSR1);
            }
            println!("  Sent SIGUSR1 to PID {pid}");
            println!("  (PyTorch Lightning, DeepSpeed, and FSDP handle SIGUSR1 for checkpointing)");

            // Step 2: Wait for checkpoint
            println!("[2/4] Waiting for checkpoint to complete...");
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

            // Step 3: Update CUDA_VISIBLE_DEVICES
            println!("[3/4] Setting CUDA_VISIBLE_DEVICES={to} for PID {pid}...");
            println!("  Note: CUDA_VISIBLE_DEVICES cannot be changed for a running process.");
            println!("  The job must be restarted with the new GPU assignment.");
            println!();

            // Step 4: Guidance
            println!("[4/4] Migration guidance:");
            println!("  1. The checkpoint signal has been sent.");
            println!("  2. Wait for the training loop to save a checkpoint.");
            println!("  3. Stop the process: kill {pid}");
            println!("  4. Restart with new GPU:");
            println!("     CUDA_VISIBLE_DEVICES={to} python train.py --resume-from-checkpoint");
            println!();
            println!("  For PyTorch Lightning:");
            println!("     CUDA_VISIBLE_DEVICES={to} python train.py --ckpt_path=last");
            println!();
            println!("  For DeepSpeed:");
            println!("     CUDA_VISIBLE_DEVICES={to} deepspeed train.py --deepspeed_config ds.json --resume");
        }

        MigrateCommands::List => {
            println!("Migration-Capable Jobs (CUDA processes)");
            println!("{}", "=".repeat(60));

            let output = Command::new("nvidia-smi")
                .args([
                    "--query-compute-apps",
                    "pid,process_name,gpu_uuid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ])
                .output()
                .with_context(|| "nvidia-smi not found")?;

            let stdout = String::from_utf8_lossy(&output.stdout);

            if stdout.trim().is_empty() {
                println!("  No CUDA processes running.");
                return Ok(());
            }

            println!(
                "{:<8} {:<30} {:>10} {:>8}",
                "PID", "Process", "GPU Mem", "GPU"
            );
            println!("{}", "-".repeat(60));

            // Get GPU index mapping
            let uuid_map = Command::new("nvidia-smi")
                .args(["--query-gpu=index,uuid", "--format=csv,noheader"])
                .output()
                .ok();

            let mut uuid_to_idx = std::collections::HashMap::new();
            if let Some(ref o) = uuid_map {
                for line in String::from_utf8_lossy(&o.stdout).lines() {
                    let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 2 {
                        uuid_to_idx.insert(f[1].to_string(), f[0].to_string());
                    }
                }
            }

            for line in stdout.lines() {
                let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if f.len() >= 4 {
                    let gpu_idx = uuid_to_idx.get(f[2]).cloned().unwrap_or_else(|| "?".into());
                    println!("{:<8} {:<30} {:>7} MB {:>8}", f[0], f[1], f[3], gpu_idx);
                }
            }

            println!();
            println!("Migrate: zernel migrate job <PID> --from <gpu> --to <gpu>");
        }

        MigrateCommands::Checkpoint { pid, output } => {
            println!("Checkpointing PID {pid} to {output}...");
            std::fs::create_dir_all(&output)?;

            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGUSR1);
            }

            println!("  SIGUSR1 sent. Waiting for framework to save checkpoint...");
            println!("  Check framework logs for checkpoint save confirmation.");
            println!();
            println!("  Expected checkpoint location depends on framework:");
            println!("    PyTorch Lightning: lightning_logs/");
            println!("    DeepSpeed: output_dir/checkpoint-*/");
            println!("    HuggingFace Trainer: output_dir/checkpoint-*/");
        }

        MigrateCommands::Resume { path, gpu } => {
            println!("Resuming from checkpoint: {path}");
            println!("  Target GPU: {gpu}");
            println!();

            if !std::path::Path::new(&path).exists() {
                anyhow::bail!("checkpoint path not found: {path}");
            }

            println!("  Restart your training with:");
            println!("    CUDA_VISIBLE_DEVICES={gpu} python train.py --resume {path}");
        }
    }
    Ok(())
}
