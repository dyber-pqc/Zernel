// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

//! GPU Memory Pressure Watchdog
//!
//! Continuously monitors GPU memory usage and sends early warnings
//! before OOM occurs. Can trigger automatic checkpointing in training
//! frameworks that support SIGUSR1 signal handling.

use std::process::Command;
use tracing::{info, warn};

/// GPU memory state.
#[derive(Debug, Clone)]
pub struct GpuMemoryState {
    pub gpu_id: u32,
    pub used_mb: u64,
    pub total_mb: u64,
    pub usage_pct: f64,
}

/// Thresholds for memory pressure alerts.
#[derive(Debug, Clone)]
pub struct WatchdogConfig {
    /// Warning threshold (percentage). Default: 85%.
    pub warn_pct: f64,
    /// Critical threshold (percentage). Default: 95%.
    pub critical_pct: f64,
    /// Polling interval in milliseconds. Default: 2000.
    pub poll_interval_ms: u64,
    /// Send SIGUSR1 to training processes at critical threshold.
    pub auto_checkpoint: bool,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            warn_pct: 85.0,
            critical_pct: 95.0,
            poll_interval_ms: 2000,
            auto_checkpoint: true,
        }
    }
}

/// Poll current GPU memory state.
pub fn poll_gpu_memory() -> Vec<GpuMemoryState> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let Ok(o) = output else { return Vec::new() };
    if !o.status.success() {
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&o.stdout);
    let mut states = Vec::new();

    for line in stdout.lines() {
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if f.len() >= 3 {
            let gpu_id: u32 = f[0].parse().unwrap_or(0);
            let used_mb: u64 = f[1].parse().unwrap_or(0);
            let total_mb: u64 = f[2].parse().unwrap_or(1);
            let usage_pct = used_mb as f64 / total_mb as f64 * 100.0;

            states.push(GpuMemoryState {
                gpu_id,
                used_mb,
                total_mb,
                usage_pct,
            });
        }
    }

    states
}

/// Get PIDs of processes using CUDA on a specific GPU.
fn get_cuda_pids(gpu_id: u32) -> Vec<u32> {
    let uuid_output = Command::new("nvidia-smi")
        .args(["--query-gpu=index,uuid", "--format=csv,noheader"])
        .output();

    let target_uuid = uuid_output
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout).lines().find_map(|line| {
                let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if f.len() >= 2 && f[0] == gpu_id.to_string() {
                    Some(f[1].to_string())
                } else {
                    None
                }
            })
        })
        .unwrap_or_default();

    if target_uuid.is_empty() {
        return Vec::new();
    }

    let proc_output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps",
            "gpu_uuid,pid",
            "--format=csv,noheader",
        ])
        .output();

    proc_output
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|line| {
                    let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 2 && f[0].contains(&target_uuid) {
                        f[1].parse().ok()
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Run the GPU memory watchdog loop.
pub async fn run_watchdog(config: WatchdogConfig) {
    info!(
        warn = config.warn_pct,
        critical = config.critical_pct,
        auto_checkpoint = config.auto_checkpoint,
        "GPU memory watchdog started"
    );

    let mut warned: std::collections::HashSet<u32> = std::collections::HashSet::new();

    loop {
        let states = poll_gpu_memory();

        for state in &states {
            if state.usage_pct >= config.critical_pct {
                warn!(
                    gpu = state.gpu_id,
                    used_mb = state.used_mb,
                    total_mb = state.total_mb,
                    usage_pct = format!("{:.1}", state.usage_pct),
                    "GPU memory CRITICAL — OOM imminent"
                );

                if config.auto_checkpoint {
                    let pids = get_cuda_pids(state.gpu_id);
                    for pid in &pids {
                        info!(
                            pid,
                            gpu = state.gpu_id,
                            "sending SIGUSR1 for emergency checkpoint"
                        );
                        #[cfg(unix)]
                        unsafe {
                            libc::kill(*pid as i32, libc::SIGUSR1);
                        }
                    }
                }
            } else if state.usage_pct >= config.warn_pct && !warned.contains(&state.gpu_id) {
                warn!(
                    gpu = state.gpu_id,
                    used_mb = state.used_mb,
                    total_mb = state.total_mb,
                    usage_pct = format!("{:.1}", state.usage_pct),
                    "GPU memory pressure warning"
                );
                warned.insert(state.gpu_id);
            } else if state.usage_pct < config.warn_pct * 0.9 {
                warned.remove(&state.gpu_id);
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(config.poll_interval_ms)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = WatchdogConfig::default();
        assert_eq!(cfg.warn_pct, 85.0);
        assert_eq!(cfg.critical_pct, 95.0);
        assert!(cfg.auto_checkpoint);
    }
}
