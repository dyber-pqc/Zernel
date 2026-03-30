// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

//! Generates simulated ML workload telemetry for development and demos.
//! Used when BPF probes are unavailable (non-Linux or no root).

use crate::aggregation::AggregatedMetrics;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

/// Simulates realistic ML telemetry data that cycles through training phases.
pub async fn run_simulator(metrics: Arc<RwLock<AggregatedMetrics>>, interval_ms: u64) {
    let mut tick = 0u64;
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));

    // Simulated GPU memory (4 GPUs, 80GB each)
    let gpu_mem_base: [u64; 4] = [
        78 * 1024 * 1024 * 1024,
        77 * 1024 * 1024 * 1024,
        79 * 1024 * 1024 * 1024,
        78 * 1024 * 1024 * 1024,
    ];

    loop {
        interval.tick().await;
        tick += 1;

        let mut m = metrics.write().await;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        m.last_update_ms = now_ms;

        // Simulate GPU memory with small fluctuations
        for (i, base) in gpu_mem_base.iter().enumerate() {
            let jitter = ((tick * (i as u64 + 1) * 7) % 500) * 1024 * 1024;
            let used = base + jitter;
            m.record_gpu_mem(1000, i as u32, used, used);
        }

        // Simulate CUDA launch latency (100-500us, with occasional spikes)
        let base_latency_ns = 142_000u64; // 142us baseline
        let spike = if tick % 50 == 0 { 500_000 } else { 0 };
        let jitter = ((tick * 31) % 200) * 1000;
        m.record_cuda_latency(1000, base_latency_ns + jitter + spike);

        // Simulate NCCL all-reduce (every ~10 ticks, 30-70ms)
        if tick % 10 == 0 {
            let duration_ns = 34_000_000 + ((tick * 17) % 30) * 1_000_000;
            m.record_nccl("all_reduce", duration_ns);
        }

        // Simulate DataLoader wait (5-15ms)
        let wait_ns = 8_000_000 + ((tick * 13) % 7) * 1_000_000;
        m.record_dataloader_wait(1000, wait_ns);

        debug!(tick, "simulator update");
    }
}
