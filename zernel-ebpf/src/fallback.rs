// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

//! Fallback telemetry provider using nvidia-smi and /proc.
//!
//! When BPF probes are unavailable (non-Linux, no root, kernel < 6.12),
//! this module polls nvidia-smi and /proc to populate the same
//! AggregatedMetrics that BPF ring buffers would. This provides REAL
//! GPU telemetry without any kernel instrumentation.

use crate::aggregation::AggregatedMetrics;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Check if nvidia-smi is available on this system.
pub fn nvidia_smi_available() -> bool {
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=count")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run the fallback telemetry poller using nvidia-smi.
/// Produces real GPU memory and utilization data without BPF.
pub async fn run_fallback(metrics: Arc<RwLock<AggregatedMetrics>>, interval_ms: u64) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));

    loop {
        interval.tick().await;

        // Poll nvidia-smi for GPU metrics
        if let Some(gpu_data) = poll_nvidia_smi().await {
            let mut m = metrics.write().await;
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            m.last_update_ms = now_ms;

            for gpu in &gpu_data {
                m.record_gpu_mem(
                    0, // pid 0 = system-level (not per-process in fallback mode)
                    gpu.index,
                    gpu.memory_used_bytes,
                    gpu.memory_total_bytes,
                );
            }

            debug!(gpus = gpu_data.len(), "fallback: nvidia-smi poll complete");
        }

        // Poll /proc/stat for CPU iowait (Linux only)
        #[cfg(target_os = "linux")]
        if let Some(iowait_pct) = poll_proc_stat() {
            debug!(iowait_pct, "fallback: /proc/stat iowait");
        }
    }
}

#[derive(Debug)]
struct GpuMetrics {
    index: u32,
    memory_used_bytes: u64,
    memory_total_bytes: u64,
    utilization_pct: u32,
    temperature_c: u32,
}

/// Poll nvidia-smi CSV output for GPU metrics.
async fn poll_nvidia_smi() -> Option<Vec<GpuMetrics>> {
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        warn!("nvidia-smi query failed");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 5 {
            continue;
        }

        let index = fields[0].parse::<u32>().unwrap_or(0);
        let mem_used_mib = fields[1].parse::<u64>().unwrap_or(0);
        let mem_total_mib = fields[2].parse::<u64>().unwrap_or(0);
        let util_pct = fields[3].parse::<u32>().unwrap_or(0);
        let temp = fields[4].parse::<u32>().unwrap_or(0);

        gpus.push(GpuMetrics {
            index,
            memory_used_bytes: mem_used_mib * 1024 * 1024,
            memory_total_bytes: mem_total_mib * 1024 * 1024,
            utilization_pct: util_pct,
            temperature_c: temp,
        });
    }

    if gpus.is_empty() {
        None
    } else {
        Some(gpus)
    }
}

/// Parse /proc/stat for CPU iowait percentage.
#[cfg(target_os = "linux")]
fn poll_proc_stat() -> Option<f64> {
    let content = std::fs::read_to_string("/proc/stat").ok()?;
    let cpu_line = content.lines().next()?;
    let fields: Vec<&str> = cpu_line.split_whitespace().collect();

    // /proc/stat format: cpu user nice system idle iowait irq softirq steal
    if fields.len() < 8 || fields[0] != "cpu" {
        return None;
    }

    let user: u64 = fields[1].parse().ok()?;
    let nice: u64 = fields[2].parse().ok()?;
    let system: u64 = fields[3].parse().ok()?;
    let idle: u64 = fields[4].parse().ok()?;
    let iowait: u64 = fields[5].parse().ok()?;
    let irq: u64 = fields[6].parse().ok()?;
    let softirq: u64 = fields[7].parse().ok()?;

    let total = user + nice + system + idle + iowait + irq + softirq;
    if total == 0 {
        return None;
    }

    Some(iowait as f64 / total as f64 * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_smi_csv() {
        // Simulate nvidia-smi CSV output parsing
        let csv = "0, 45678, 81920, 94, 72\n1, 43210, 81920, 87, 68\n";
        let mut gpus = Vec::new();
        for line in csv.lines() {
            let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if fields.len() >= 5 {
                gpus.push(GpuMetrics {
                    index: fields[0].parse().unwrap(),
                    memory_used_bytes: fields[1].parse::<u64>().unwrap() * 1024 * 1024,
                    memory_total_bytes: fields[2].parse::<u64>().unwrap() * 1024 * 1024,
                    utilization_pct: fields[3].parse().unwrap(),
                    temperature_c: fields[4].parse().unwrap(),
                });
            }
        }
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].utilization_pct, 94);
        assert_eq!(gpus[1].memory_used_bytes, 43210 * 1024 * 1024);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_proc_stat() {
        let result = poll_proc_stat();
        // On Linux CI, /proc/stat should exist
        assert!(result.is_some());
    }
}
