// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;
use std::collections::HashMap;

/// Aggregated metrics computed from raw BPF events.
/// Exported to Prometheus and streamed to CLI via WebSocket.
#[derive(Debug, Default, Serialize)]
pub struct AggregatedMetrics {
    /// GPU memory usage per (pid, gpu_id).
    pub gpu_memory: HashMap<(u32, u32), GpuMemMetrics>,
    /// CUDA launch latency histogram per pid.
    pub cuda_latency: HashMap<u32, LatencyHistogram>,
    /// NCCL collective duration per operation type.
    pub nccl_duration: HashMap<String, LatencyHistogram>,
}

#[derive(Debug, Default, Serialize)]
pub struct GpuMemMetrics {
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub alloc_count: u64,
    pub free_count: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct LatencyHistogram {
    pub count: u64,
    pub sum_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub p50_ns: u64,
    pub p99_ns: u64,
    samples: Vec<u64>,
}

impl LatencyHistogram {
    pub fn record(&mut self, value_ns: u64) {
        self.count += 1;
        self.sum_ns += value_ns;
        if self.count == 1 || value_ns < self.min_ns {
            self.min_ns = value_ns;
        }
        if value_ns > self.max_ns {
            self.max_ns = value_ns;
        }
        self.samples.push(value_ns);

        // Recompute percentiles periodically (every 100 samples).
        if self.samples.len() % 100 == 0 {
            self.recompute_percentiles();
        }
    }

    fn recompute_percentiles(&mut self) {
        if self.samples.is_empty() {
            return;
        }
        self.samples.sort_unstable();
        let len = self.samples.len();
        self.p50_ns = self.samples[len / 2];
        self.p99_ns = self.samples[(len as f64 * 0.99) as usize];
    }
}
