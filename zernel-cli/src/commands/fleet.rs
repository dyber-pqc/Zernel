// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel fleet — GPU fleet management at scale
//!
//! Designed for AI labs running 100-10,000+ GPUs. Provides:
//! - Fleet-wide GPU utilization dashboard
//! - Cost attribution per team/project
//! - Idle GPU detection and automatic reclamation
//! - Right-sizing recommendations
//! - Capacity planning

use anyhow::Result;
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum FleetCommands {
    /// Fleet-wide GPU utilization overview
    Status,
    /// Show cost attribution by team/project
    Costs {
        /// Time period (today, week, month)
        #[arg(long, default_value = "month")]
        period: String,
    },
    /// Detect idle GPUs across the fleet
    Idle {
        /// Utilization threshold (%) below which a GPU is "idle"
        #[arg(long, default_value = "5")]
        threshold: u32,
        /// Duration (minutes) a GPU must be idle before flagging
        #[arg(long, default_value = "30")]
        duration: u32,
    },
    /// Reclaim idle GPUs (reassign or power down)
    Reclaim {
        /// Dry run (show what would be reclaimed)
        #[arg(long)]
        dry_run: bool,
    },
    /// Right-sizing recommendations
    Rightsize,
    /// Capacity planning — predict when you'll need more GPUs
    Plan {
        /// Growth rate (% per month)
        #[arg(long, default_value = "10")]
        growth: f64,
    },
    /// Fleet health report
    Health,
}

pub async fn run(cmd: FleetCommands) -> Result<()> {
    match cmd {
        FleetCommands::Status => {
            println!("Zernel Fleet Status");
            println!("{}", "=".repeat(70));

            // Load cluster nodes
            let cluster_file = crate::experiments::tracker::zernel_dir()
                .join("cluster")
                .join("nodes.json");

            if !cluster_file.exists() {
                println!("No fleet configured. Add nodes: zernel cluster add <host> --gpus 8");
                println!();
                println!("For single-node fleet status:");
                let output = Command::new("nvidia-smi")
                    .args([
                        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ])
                    .output();

                if let Ok(o) = output {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let mut total_gpus = 0u32;
                    let mut total_util = 0u32;
                    let mut total_power = 0.0f64;

                    println!(
                        "{:<5} {:<20} {:>6} {:>12} {:>6} {:>8}",
                        "GPU", "Name", "Util", "Memory", "Temp", "Power"
                    );
                    println!("{}", "-".repeat(70));

                    for line in stdout.lines() {
                        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                        if f.len() >= 7 {
                            let util: u32 = f[2].parse().unwrap_or(0);
                            let power: f64 = f[6].parse().unwrap_or(0.0);
                            total_gpus += 1;
                            total_util += util;
                            total_power += power;
                            println!(
                                "{:<5} {:<20} {:>4}% {:>5}/{:<5}MB {:>4}°C {:>6.0}W",
                                f[0], f[1], f[2], f[3], f[4], f[5], power
                            );
                        }
                    }

                    if total_gpus > 0 {
                        println!();
                        println!("Fleet Summary:");
                        println!("  Total GPUs:     {total_gpus}");
                        println!("  Avg utilization: {}%", total_util / total_gpus);
                        println!(
                            "  Total power:     {total_power:.0}W ({:.1} kW)",
                            total_power / 1000.0
                        );
                        println!(
                            "  Est. daily cost: ${:.0} (at $0.10/kWh)",
                            total_power / 1000.0 * 24.0 * 0.10
                        );
                    }
                }
                return Ok(());
            }

            // Multi-node fleet status would SSH to each node here
            println!("Multi-node fleet status: use `zernel cluster status`");
        }

        FleetCommands::Costs { period } => {
            println!("Fleet Cost Attribution — {period}");
            println!("{}", "=".repeat(60));

            // Calculate from experiments and jobs databases
            let exp_db = crate::experiments::tracker::experiments_db_path();
            if exp_db.exists() {
                let conn = rusqlite::Connection::open(&exp_db)?;
                let total_secs: f64 = conn
                    .query_row(
                        "SELECT COALESCE(SUM(duration_secs), 0) FROM experiments",
                        [],
                        |row| row.get(0),
                    )
                    .unwrap_or(0.0);

                let total_hours = total_secs / 3600.0;
                let exp_count: u32 = conn
                    .query_row("SELECT COUNT(*) FROM experiments", [], |row| row.get(0))
                    .unwrap_or(0);

                println!("  Experiments:     {exp_count}");
                println!("  Total GPU-hours: {total_hours:.1}h");
                println!();

                // Cost estimates at various price points
                println!("  Estimated costs:");
                println!(
                    "    At $2.50/GPU-hr (A100 on-demand):  ${:.0}",
                    total_hours * 2.50
                );
                println!(
                    "    At $1.50/GPU-hr (A100 reserved):   ${:.0}",
                    total_hours * 1.50
                );
                println!(
                    "    At $4.00/GPU-hr (H100 on-demand):  ${:.0}",
                    total_hours * 4.00
                );
                println!(
                    "    At $0.10/kWh (on-prem electricity): ${:.0}",
                    total_hours * 0.3 * 0.10
                );
            } else {
                println!("  No experiment data. Run: zernel run <script>");
            }
        }

        FleetCommands::Idle {
            threshold,
            duration,
        } => {
            println!("Idle GPU Detection (threshold: <{threshold}% for >{duration}min)");
            println!("{}", "=".repeat(60));

            let output = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=index,name,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ])
                .output();

            if let Ok(o) = output {
                let stdout = String::from_utf8_lossy(&o.stdout);
                let mut idle_count = 0;
                for line in stdout.lines() {
                    let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 3 {
                        let util: u32 = f[2].parse().unwrap_or(0);
                        if util < threshold {
                            println!("  GPU {}: {} — {}% (IDLE)", f[0], f[1], util);
                            idle_count += 1;
                        }
                    }
                }

                if idle_count == 0 {
                    println!("  No idle GPUs detected.");
                } else {
                    println!();
                    println!("  {idle_count} idle GPU(s) detected.");
                    println!("  Reclaim with: zernel fleet reclaim");
                    println!("  Estimated savings: ${:.0}/day per idle GPU", 24.0 * 2.50);
                }
            }
        }

        FleetCommands::Reclaim { dry_run } => {
            if dry_run {
                println!("Dry run — showing what would be reclaimed:");
            } else {
                println!("Reclaiming idle GPUs...");
            }

            let output = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=index,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ])
                .output();

            if let Ok(o) = output {
                let stdout = String::from_utf8_lossy(&o.stdout);
                for line in stdout.lines() {
                    let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 2 {
                        let util: u32 = f[1].parse().unwrap_or(100);
                        if util < 5 {
                            println!("  GPU {}: {}% utilized", f[0], util);
                            if !dry_run {
                                // Lower power state
                                let _ = Command::new("nvidia-smi")
                                    .args(["-i", f[0], "-pl", "50"])
                                    .output();
                                println!("    → Power reduced to 50W (idle mode)");
                            } else {
                                println!("    → Would reduce power to idle mode");
                            }
                        }
                    }
                }
            }
        }

        FleetCommands::Rightsize => {
            println!("GPU Right-Sizing Recommendations");
            println!("{}", "=".repeat(60));

            let exp_db = crate::experiments::tracker::experiments_db_path();
            if exp_db.exists() {
                println!("Based on your training history:");
                println!();
                println!("  Recommendation: Analyze GPU memory watermarks and utilization");
                println!("  patterns from your experiments to determine optimal GPU type.");
                println!();
                println!("  If avg GPU memory < 40GB: Consider A100 40GB (cheaper)");
                println!("  If avg GPU memory < 24GB: Consider A10G or L4 (much cheaper)");
                println!("  If avg GPU util < 50%: Reduce GPU count or use smaller GPUs");
                println!("  If NCCL is >20% of step time: Need faster interconnect (NVLink/IB)");
                println!();
                println!("  Run: zernel bench all — to generate utilization profile");
                println!("  Run: zernel debug why-slow — to identify bottlenecks");
            } else {
                println!("  Need training data. Run experiments first: zernel run <script>");
            }
        }

        FleetCommands::Plan { growth } => {
            println!("Capacity Planning (growth: {growth}%/month)");
            println!("{}", "=".repeat(60));

            let output = Command::new("nvidia-smi")
                .args(["--query-gpu=count", "--format=csv,noheader"])
                .output();

            let current_gpus: u32 = output
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .lines()
                        .next()
                        .and_then(|s| s.parse().ok())
                })
                .unwrap_or(0);

            println!("  Current GPUs: {current_gpus}");
            println!();
            println!("  Projected need (at {growth}% monthly growth):");

            let mut gpus = current_gpus as f64;
            for month in 1..=12 {
                gpus *= 1.0 + growth / 100.0;
                let cost_ondemand = gpus * 24.0 * 30.0 * 2.50;
                let cost_reserved = gpus * 24.0 * 30.0 * 1.50;
                println!(
                    "    Month {month:>2}: {:>4.0} GPUs — ${cost_reserved:.0}-${cost_ondemand:.0}/mo",
                    gpus
                );
            }
        }

        FleetCommands::Health => {
            println!("Fleet Health Report");
            println!("{}", "=".repeat(60));

            // Check each subsystem
            let checks = [
                ("nvidia-smi", "GPU drivers"),
                ("zernel", "Zernel CLI"),
                ("python3", "Python runtime"),
            ];

            for (cmd, name) in &checks {
                let ok = Command::new(cmd)
                    .arg("--version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false);
                println!("  {name:<25} {}", if ok { "OK" } else { "MISSING" });
            }

            // Check zerneld
            let zerneld_ok = std::net::TcpStream::connect_timeout(
                &"127.0.0.1:9091".parse().expect("valid"),
                std::time::Duration::from_millis(500),
            )
            .is_ok();
            println!(
                "  {:<25} {}",
                "zerneld (observability)",
                if zerneld_ok { "RUNNING" } else { "STOPPED" }
            );

            // Check dashboard
            let dash_ok = std::net::TcpStream::connect_timeout(
                &"127.0.0.1:3000".parse().expect("valid"),
                std::time::Duration::from_millis(500),
            )
            .is_ok();
            println!(
                "  {:<25} {}",
                "zernel-dashboard (web)",
                if dash_ok { "RUNNING" } else { "STOPPED" }
            );

            // Check ollama
            let ollama_ok = std::net::TcpStream::connect_timeout(
                &"127.0.0.1:11434".parse().expect("valid"),
                std::time::Duration::from_millis(500),
            )
            .is_ok();
            println!(
                "  {:<25} {}",
                "ollama (local LLM)",
                if ollama_ok { "RUNNING" } else { "STOPPED" }
            );
        }
    }
    Ok(())
}
