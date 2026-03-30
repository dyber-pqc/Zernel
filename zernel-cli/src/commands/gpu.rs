// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel gpu — GPU management CLI (nvidia-smi replacement)

use anyhow::{Context, Result};
use clap::Subcommand;

use std::process::Command;

#[derive(Subcommand)]
pub enum GpuCommands {
    /// Show clean GPU overview (default if no subcommand)
    Status,
    /// Real-time GPU process viewer (like htop for GPUs)
    Top,
    /// Show GPU memory usage by process
    Mem,
    /// Kill all processes on a GPU
    Kill {
        /// GPU index (0, 1, 2, ...)
        gpu: u32,
    },
    /// Reserve GPUs for exclusive use
    Lock {
        /// GPU indices (comma-separated: 0,1,2)
        gpus: String,
        /// Job or user to lock for
        #[arg(long, rename_all = "kebab-case")]
        for_job: Option<String>,
    },
    /// Release GPU reservation
    Unlock {
        /// GPU indices (comma-separated)
        gpus: String,
    },
    /// Monitor GPU temperature with alerts
    Temp {
        /// Alert threshold in Celsius
        #[arg(long, default_value = "85")]
        alert: u32,
    },
    /// Set GPU power limits
    Power {
        /// Power limit (e.g., 300W)
        #[arg(long)]
        limit: Option<String>,
    },
    /// GPU health check (ECC errors, throttling, PCIe)
    Health,
}

fn query_nvidia_smi(fields: &str) -> Result<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu", fields, "--format=csv,noheader,nounits"])
        .output()
        .with_context(|| "nvidia-smi not found — is the NVIDIA driver installed?")?;
    if !output.status.success() {
        anyhow::bail!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn query_nvidia_smi_procs() -> Result<String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps",
            "gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .with_context(|| "nvidia-smi not found")?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub async fn run(cmd: GpuCommands) -> Result<()> {
    match cmd {
        GpuCommands::Status => {
            let data = query_nvidia_smi(
                "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            )?;

            println!("Zernel GPU Status");
            println!("{}", "=".repeat(80));
            println!(
                "{:<5} {:<22} {:>5} {:>12} {:>6} {:>8} {:>10}",
                "GPU", "Name", "Util", "Memory", "Temp", "Power", "Limit"
            );
            println!("{}", "-".repeat(80));

            for line in data.lines() {
                let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if f.len() >= 8 {
                    let util_pct: u32 = f[2].parse().unwrap_or(0);
                    let bar = gpu_bar(util_pct, 10);
                    println!(
                        "{:<5} {:<22} {} {:>5}/{:<5} MB {:>3}°C {:>6}W/{:<4}W",
                        f[0], f[1], bar, f[3], f[4], f[5], f[6], f[7]
                    );
                }
            }
            println!();
        }

        GpuCommands::Top => {
            println!("Zernel GPU Top — Press Ctrl+C to exit");
            println!();
            loop {
                // Clear screen
                print!("\x1B[2J\x1B[H");

                let gpu_data = query_nvidia_smi(
                    "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                )?;
                let proc_data = query_nvidia_smi_procs()?;

                println!("Zernel GPU Top");
                println!("{}", "=".repeat(80));

                for line in gpu_data.lines() {
                    let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 6 {
                        let util: u32 = f[2].parse().unwrap_or(0);
                        println!(
                            "GPU {} ({}) {} {}/{}MB {}°C",
                            f[0],
                            f[1],
                            gpu_bar(util, 20),
                            f[3],
                            f[4],
                            f[5]
                        );
                    }
                }

                println!();
                println!("{:<8} {:<30} {:>10}", "PID", "Process", "GPU Mem (MB)");
                println!("{}", "-".repeat(50));

                if proc_data.trim().is_empty() {
                    println!("  (no GPU processes running)");
                } else {
                    for line in proc_data.lines() {
                        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                        if f.len() >= 4 {
                            println!("{:<8} {:<30} {:>10}", f[1], f[2], f[3]);
                        }
                    }
                }

                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            }
        }

        GpuCommands::Mem => {
            let proc_data = query_nvidia_smi_procs()?;

            println!("GPU Memory by Process");
            println!("{}", "=".repeat(60));
            println!(
                "{:<8} {:<30} {:>10} {:>8}",
                "PID", "Process", "GPU Mem", "GPU"
            );
            println!("{}", "-".repeat(60));

            if proc_data.trim().is_empty() {
                println!("  (no GPU processes running)");
            } else {
                let mut total: u64 = 0;
                for line in proc_data.lines() {
                    let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 4 {
                        let mem: u64 = f[3].parse().unwrap_or(0);
                        total += mem;
                        println!("{:<8} {:<30} {:>7} MB {:>8}", f[1], f[2], f[3], f[0]);
                    }
                }
                println!("{}", "-".repeat(60));
                println!("{:<38} {:>7} MB", "TOTAL", total);
            }
        }

        GpuCommands::Kill { gpu } => {
            let proc_data = query_nvidia_smi_procs()?;
            let gpu_str = gpu.to_string();
            let mut killed = 0;

            // Get GPU UUID for this index
            let uuid_data = query_nvidia_smi("index,uuid")?;
            let target_uuid: Option<String> = uuid_data.lines().find_map(|line| {
                let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if f.len() >= 2 && f[0] == gpu_str {
                    Some(f[1].to_string())
                } else {
                    None
                }
            });

            let Some(uuid) = target_uuid else {
                println!("GPU {gpu} not found");
                return Ok(());
            };

            for line in proc_data.lines() {
                let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if f.len() >= 2 && f[0].contains(&uuid) {
                    if let Ok(pid) = f[1].parse::<u32>() {
                        println!("Killing PID {pid} ({})...", f.get(2).unwrap_or(&""));
                        #[cfg(unix)]
                        unsafe {
                            libc::kill(pid as i32, libc::SIGTERM);
                        }
                        #[cfg(not(unix))]
                        {
                            let _ = Command::new("taskkill")
                                .args(["/PID", &pid.to_string(), "/F"])
                                .output();
                        }
                        killed += 1;
                    }
                }
            }

            if killed == 0 {
                println!("No processes found on GPU {gpu}");
            } else {
                println!("Killed {killed} process(es) on GPU {gpu}");
            }
        }

        GpuCommands::Lock { gpus, for_job } => {
            let job_name = for_job.unwrap_or_else(|| "manual".into());
            // Write lock file
            let lock_dir = crate::experiments::tracker::zernel_dir().join("gpu-locks");
            std::fs::create_dir_all(&lock_dir)?;
            for gpu in gpus.split(',') {
                let gpu = gpu.trim();
                let lock_file = lock_dir.join(format!("gpu-{gpu}.lock"));
                std::fs::write(&lock_file, &job_name)?;
                println!("GPU {gpu} locked for: {job_name}");
            }
            println!("Set CUDA_VISIBLE_DEVICES={gpus} in your training script.");
        }

        GpuCommands::Unlock { gpus } => {
            let lock_dir = crate::experiments::tracker::zernel_dir().join("gpu-locks");
            for gpu in gpus.split(',') {
                let gpu = gpu.trim();
                let lock_file = lock_dir.join(format!("gpu-{gpu}.lock"));
                if lock_file.exists() {
                    std::fs::remove_file(&lock_file)?;
                    println!("GPU {gpu} unlocked");
                } else {
                    println!("GPU {gpu} was not locked");
                }
            }
        }

        GpuCommands::Temp { alert } => {
            println!("Monitoring GPU temperatures (alert at {alert}°C)...");
            loop {
                let data = query_nvidia_smi("index,temperature.gpu")?;
                for line in data.lines() {
                    let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 2 {
                        let temp: u32 = f[1].parse().unwrap_or(0);
                        let indicator = if temp >= alert { "ALERT" } else { "ok" };
                        println!("GPU {}: {}°C [{}]", f[0], temp, indicator);
                    }
                }
                println!();
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        }

        GpuCommands::Power { limit } => {
            if let Some(limit_str) = limit {
                let watts = limit_str.trim_end_matches('W').trim_end_matches('w');
                let status = Command::new("nvidia-smi").args(["-pl", watts]).status()?;
                if status.success() {
                    println!("Power limit set to {watts}W across all GPUs");
                } else {
                    println!("Failed to set power limit (requires root)");
                }
            } else {
                let data = query_nvidia_smi("index,power.draw,power.limit,power.max_limit")?;
                println!("GPU Power Status");
                println!("{:<5} {:>10} {:>10} {:>10}", "GPU", "Draw", "Limit", "Max");
                for line in data.lines() {
                    let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 4 {
                        println!("{:<5} {:>8}W {:>8}W {:>8}W", f[0], f[1], f[2], f[3]);
                    }
                }
            }
        }

        GpuCommands::Health => {
            println!("Zernel GPU Health Check");
            println!("{}", "=".repeat(60));

            let data = query_nvidia_smi(
                "index,name,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,clocks_throttle_reasons.active,pcie.link.gen.current,pcie.link.width.current",
            )?;

            for line in data.lines() {
                let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if f.len() >= 7 {
                    println!("GPU {} ({})", f[0], f[1]);
                    let ecc_corr = f[2];
                    let ecc_uncorr = f[3];
                    let throttle = f[4];
                    let pcie_gen = f[5];
                    let pcie_width = f[6];

                    let ecc_status = if ecc_uncorr != "0" && ecc_uncorr != "N/A" {
                        "FAIL — uncorrected ECC errors"
                    } else if ecc_corr != "0" && ecc_corr != "N/A" {
                        "WARN — corrected ECC errors"
                    } else {
                        "OK"
                    };

                    let throttle_status =
                        if throttle.contains("0x") && throttle != "0x0000000000000000" {
                            "WARN — throttling active"
                        } else {
                            "OK"
                        };

                    println!("  ECC:        {ecc_status}");
                    println!("  Throttling: {throttle_status}");
                    println!("  PCIe:       Gen{pcie_gen} x{pcie_width}");
                    println!();
                }
            }
        }
    }
    Ok(())
}

fn gpu_bar(pct: u32, width: usize) -> String {
    let filled = (pct as usize * width) / 100;
    let empty = width.saturating_sub(filled);
    let color = if pct > 90 {
        "\x1b[32m"
    } else if pct > 70 {
        "\x1b[33m"
    } else {
        "\x1b[31m"
    };
    format!(
        "{color}[{}{}]\x1b[0m {:>3}%",
        "#".repeat(filled),
        " ".repeat(empty),
        pct
    )
}
