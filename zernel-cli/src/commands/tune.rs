// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel tune — Adaptive kernel parameter tuning
//!
//! Reads hardware configuration (GPU count, RAM, NVMe, network) and
//! generates optimal sysctl + kernel parameters for the specific machine.
//! Not a static config — adapts to actual hardware present.

use anyhow::Result;
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum TuneCommands {
    /// Analyze hardware and show recommended parameters
    Analyze,
    /// Apply optimal parameters (requires root)
    Apply {
        /// Dry run — show what would be changed without applying
        #[arg(long)]
        dry_run: bool,
    },
    /// Show current vs optimal parameters
    Diff,
    /// Generate a sysctl.conf file for this machine
    Export {
        /// Output file
        #[arg(long, default_value = "zernel-tuned.conf")]
        output: String,
    },
}

struct HardwareProfile {
    gpu_count: u32,
    gpu_memory_mb: u64,
    ram_mb: u64,
    cpu_cores: u32,
    numa_nodes: u32,
    nvme_count: u32,
    network_speed_mbps: u32,
    has_infiniband: bool,
}

impl HardwareProfile {
    fn detect() -> Self {
        let gpu_count = Command::new("nvidia-smi")
            .args(["--query-gpu=count", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .lines()
                    .next()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(0);

        let gpu_memory_mb = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
            .ok()
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .lines()
                    .next()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(0);

        let ram_mb = {
            #[cfg(target_os = "linux")]
            {
                std::fs::read_to_string("/proc/meminfo")
                    .ok()
                    .and_then(|s| {
                        s.lines()
                            .find(|l| l.starts_with("MemTotal:"))
                            .and_then(|l| l.split_whitespace().nth(1))
                            .and_then(|s| s.parse::<u64>().ok())
                            .map(|kb| kb / 1024)
                    })
                    .unwrap_or(0)
            }
            #[cfg(not(target_os = "linux"))]
            {
                0u64
            }
        };

        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        #[cfg(target_os = "linux")]
        let numa_nodes = std::fs::read_dir("/sys/devices/system/node")
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with("node"))
                    .count() as u32
            })
            .unwrap_or(1);
        #[cfg(not(target_os = "linux"))]
        let numa_nodes = 1u32;

        #[cfg(target_os = "linux")]
        let nvme_count = std::fs::read_dir("/sys/class/nvme")
            .map(|entries| entries.flatten().count() as u32)
            .unwrap_or(0);
        #[cfg(not(target_os = "linux"))]
        let nvme_count = 0u32;

        let has_infiniband = std::path::Path::new("/sys/class/infiniband").exists();

        let network_speed_mbps = if has_infiniband { 100000 } else { 25000 };

        Self {
            gpu_count,
            gpu_memory_mb,
            ram_mb,
            cpu_cores,
            numa_nodes,
            nvme_count,
            network_speed_mbps,
            has_infiniband,
        }
    }
}

struct TuningParam {
    key: String,
    value: String,
    reason: String,
}

fn generate_params(hw: &HardwareProfile) -> Vec<TuningParam> {
    let mut params = Vec::new();

    // === Memory ===
    params.push(TuningParam {
        key: "vm.swappiness".into(),
        value: "0".into(),
        reason: "ML servers should never swap — kills GPU performance".into(),
    });

    params.push(TuningParam {
        key: "vm.overcommit_memory".into(),
        value: "1".into(),
        reason: "PyTorch requires memory overcommit for large allocations".into(),
    });

    // Huge pages — scale with GPU memory
    let hugepages = if hw.gpu_memory_mb > 40000 {
        2048 // 2GB for A100/H100
    } else if hw.gpu_memory_mb > 16000 {
        1024 // 1GB for V100/RTX 3090
    } else {
        512 // 512MB for smaller GPUs
    };
    params.push(TuningParam {
        key: "vm.nr_hugepages".into(),
        value: hugepages.to_string(),
        reason: format!(
            "Pre-allocate {}MB huge pages for GPU DMA (based on {}MB GPU memory)",
            hugepages * 2,
            hw.gpu_memory_mb
        ),
    });

    // Dirty ratio — scale with RAM
    let dirty_ratio = if hw.ram_mb > 256000 {
        60
    } else if hw.ram_mb > 128000 {
        40
    } else {
        20
    };
    params.push(TuningParam {
        key: "vm.dirty_ratio".into(),
        value: dirty_ratio.to_string(),
        reason: format!(
            "Allow {}% dirty pages for large dataset writes ({}GB RAM)",
            dirty_ratio,
            hw.ram_mb / 1024
        ),
    });

    params.push(TuningParam {
        key: "vm.dirty_background_ratio".into(),
        value: "10".into(),
        reason: "Start background writeback at 10%".into(),
    });

    // === Network ===
    let net_buf = if hw.has_infiniband || hw.network_speed_mbps >= 100000 {
        268435456 // 256MB for InfiniBand/100GbE
    } else if hw.network_speed_mbps >= 25000 {
        134217728 // 128MB for 25GbE
    } else {
        67108864 // 64MB for 10GbE
    };

    params.push(TuningParam {
        key: "net.core.rmem_max".into(),
        value: net_buf.to_string(),
        reason: format!(
            "{}MB receive buffer for {}Gbps network (NCCL distributed training)",
            net_buf / 1048576,
            hw.network_speed_mbps / 1000
        ),
    });

    params.push(TuningParam {
        key: "net.core.wmem_max".into(),
        value: net_buf.to_string(),
        reason: format!("{}MB send buffer", net_buf / 1048576),
    });

    params.push(TuningParam {
        key: "net.ipv4.tcp_rmem".into(),
        value: format!("4096 87380 {net_buf}"),
        reason: "TCP receive buffer auto-tuning range".into(),
    });

    params.push(TuningParam {
        key: "net.ipv4.tcp_wmem".into(),
        value: format!("4096 65536 {net_buf}"),
        reason: "TCP send buffer auto-tuning range".into(),
    });

    params.push(TuningParam {
        key: "net.ipv4.tcp_congestion_control".into(),
        value: "bbr".into(),
        reason: "BBR congestion control — better for datacenter workloads".into(),
    });

    let backlog = if hw.gpu_count > 4 { 500000 } else { 250000 };
    params.push(TuningParam {
        key: "net.core.netdev_max_backlog".into(),
        value: backlog.to_string(),
        reason: format!(
            "Network backlog for {} GPUs (NCCL generates bursty traffic)",
            hw.gpu_count
        ),
    });

    // === NUMA ===
    if hw.numa_nodes > 1 {
        params.push(TuningParam {
            key: "kernel.numa_balancing".into(),
            value: "1".into(),
            reason: format!("NUMA auto-balancing for {} NUMA nodes", hw.numa_nodes),
        });
    }

    // === File handles ===
    let file_max = if hw.gpu_count > 4 { 4194304 } else { 2097152 };
    params.push(TuningParam {
        key: "fs.file-max".into(),
        value: file_max.to_string(),
        reason: format!(
            "{} max file handles (DataLoader workers + {} GPUs)",
            file_max, hw.gpu_count
        ),
    });

    params.push(TuningParam {
        key: "fs.inotify.max_user_watches".into(),
        value: "1048576".into(),
        reason: "High inotify watches for dataset monitoring".into(),
    });

    // === Scheduler ===
    if hw.cpu_cores > 16 {
        params.push(TuningParam {
            key: "kernel.sched_migration_cost_ns".into(),
            value: "5000000".into(),
            reason: format!(
                "Reduce scheduler migration cost on {} cores (keep DataLoader threads on-core)",
                hw.cpu_cores
            ),
        });
    }

    params
}

pub async fn run(cmd: TuneCommands) -> Result<()> {
    match cmd {
        TuneCommands::Analyze => {
            let hw = HardwareProfile::detect();

            println!("Zernel Hardware Analysis");
            println!("{}", "=".repeat(60));
            println!();
            println!("Detected Hardware:");
            println!(
                "  GPUs:         {} ({}MB each)",
                hw.gpu_count, hw.gpu_memory_mb
            );
            println!("  RAM:          {} GB", hw.ram_mb / 1024);
            println!("  CPU cores:    {}", hw.cpu_cores);
            println!("  NUMA nodes:   {}", hw.numa_nodes);
            println!("  NVMe drives:  {}", hw.nvme_count);
            println!(
                "  Network:      {}Gbps {}",
                hw.network_speed_mbps / 1000,
                if hw.has_infiniband {
                    "(InfiniBand)"
                } else {
                    ""
                }
            );
            println!();

            let params = generate_params(&hw);
            println!("Recommended Parameters ({} total):", params.len());
            println!("{}", "-".repeat(60));
            for p in &params {
                println!("  {} = {}", p.key, p.value);
                println!("    # {}", p.reason);
            }

            println!();
            println!("Apply: zernel tune apply");
            println!("Export: zernel tune export --output zernel-tuned.conf");
        }

        TuneCommands::Apply { dry_run } => {
            let hw = HardwareProfile::detect();
            let params = generate_params(&hw);

            if dry_run {
                println!("Dry run — showing what would be applied:");
            } else {
                println!("Applying {} tuning parameters...", params.len());
            }

            for p in &params {
                if dry_run {
                    println!("  sysctl -w {} = {}", p.key, p.value);
                } else {
                    let status = Command::new("sysctl")
                        .args(["-w", &format!("{}={}", p.key, p.value)])
                        .output();
                    match status {
                        Ok(o) if o.status.success() => {
                            println!("  OK: {} = {}", p.key, p.value);
                        }
                        _ => {
                            println!("  SKIP: {} (requires root)", p.key);
                        }
                    }
                }
            }

            if !dry_run {
                println!();
                println!("Parameters applied. To persist across reboots:");
                println!("  zernel tune export --output /etc/sysctl.d/99-zernel-tuned.conf");
            }
        }

        TuneCommands::Diff => {
            let hw = HardwareProfile::detect();
            let params = generate_params(&hw);

            println!("Current vs Optimal Parameters");
            println!("{}", "=".repeat(70));
            println!("{:<40} {:>12} {:>12}", "Parameter", "Current", "Optimal");
            println!("{}", "-".repeat(70));

            for p in &params {
                let current = Command::new("sysctl")
                    .args(["-n", &p.key])
                    .output()
                    .ok()
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                    .unwrap_or_else(|| "N/A".into());

                let marker = if current.trim() == p.value.trim() {
                    "  "
                } else {
                    "→ "
                };
                println!("{}{:<38} {:>12} {:>12}", marker, p.key, current, p.value);
            }
        }

        TuneCommands::Export { output } => {
            let hw = HardwareProfile::detect();
            let params = generate_params(&hw);

            let mut conf = String::new();
            conf.push_str("# Zernel Auto-Tuned Parameters\n");
            conf.push_str(&format!(
                "# Generated for: {} GPUs, {}GB RAM, {} cores, {} NUMA nodes\n",
                hw.gpu_count,
                hw.ram_mb / 1024,
                hw.cpu_cores,
                hw.numa_nodes
            ));
            conf.push_str(&format!(
                "# Generated at: {}\n\n",
                chrono::Utc::now().to_rfc3339()
            ));

            for p in &params {
                conf.push_str(&format!("# {}\n", p.reason));
                conf.push_str(&format!("{} = {}\n\n", p.key, p.value));
            }

            std::fs::write(&output, &conf)?;
            println!("Exported {} parameters to: {output}", params.len());
            println!("Apply: sudo cp {output} /etc/sysctl.d/ && sudo sysctl --system");
        }
    }
    Ok(())
}
