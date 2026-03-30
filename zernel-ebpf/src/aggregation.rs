// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;
use std::collections::HashMap;

/// Aggregated metrics computed from raw BPF events.
#[derive(Debug, Default, Clone, Serialize)]
pub struct AggregatedMetrics {
    pub gpu_memory: HashMap<String, GpuMemMetrics>,
    pub cuda_latency: HashMap<u32, LatencyHistogram>,
    pub nccl_duration: HashMap<String, LatencyHistogram>,
    pub dataloader_wait: HashMap<u32, LatencyHistogram>,
    pub dist_sync: HashMap<u32, LatencyHistogram>,
    /// Timestamp (ms since epoch) of last update.
    pub last_update_ms: u64,
}

impl AggregatedMetrics {
    /// Record a GPU memory event.
    pub fn record_gpu_mem(&mut self, pid: u32, gpu_id: u32, used: u64, peak: u64) {
        let key = format!("{pid}:{gpu_id}");
        let entry = self.gpu_memory.entry(key).or_default();
        entry.current_bytes = used;
        if peak > entry.peak_bytes {
            entry.peak_bytes = peak;
        }
        entry.alloc_count += 1;
    }

    /// Record a CUDA kernel launch latency.
    pub fn record_cuda_latency(&mut self, pid: u32, latency_ns: u64) {
        self.cuda_latency.entry(pid).or_default().record(latency_ns);
    }

    /// Record an NCCL collective duration.
    pub fn record_nccl(&mut self, op: &str, duration_ns: u64) {
        self.nccl_duration
            .entry(op.to_string())
            .or_default()
            .record(duration_ns);
    }

    /// Record a DataLoader wait time.
    pub fn record_dataloader_wait(&mut self, pid: u32, wait_ns: u64) {
        self.dataloader_wait.entry(pid).or_default().record(wait_ns);
    }

    /// Format as Prometheus text exposition.
    pub fn to_prometheus(&self) -> String {
        let mut out = String::with_capacity(4096);

        // GPU Memory
        out.push_str("# HELP zernel_gpu_memory_used_bytes Current GPU memory usage\n");
        out.push_str("# TYPE zernel_gpu_memory_used_bytes gauge\n");
        for (key, m) in &self.gpu_memory {
            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() == 2 {
                out.push_str(&format!(
                    "zernel_gpu_memory_used_bytes{{pid=\"{}\",gpu_id=\"{}\"}} {}\n",
                    parts[0], parts[1], m.current_bytes
                ));
            }
        }

        out.push_str("# HELP zernel_gpu_memory_peak_bytes Peak GPU memory usage\n");
        out.push_str("# TYPE zernel_gpu_memory_peak_bytes gauge\n");
        for (key, m) in &self.gpu_memory {
            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() == 2 {
                out.push_str(&format!(
                    "zernel_gpu_memory_peak_bytes{{pid=\"{}\",gpu_id=\"{}\"}} {}\n",
                    parts[0], parts[1], m.peak_bytes
                ));
            }
        }

        // CUDA Latency
        out.push_str("# HELP zernel_cuda_launch_latency_seconds CUDA kernel launch latency\n");
        out.push_str("# TYPE zernel_cuda_launch_latency_seconds summary\n");
        for (pid, h) in &self.cuda_latency {
            out.push_str(&format!(
                "zernel_cuda_launch_latency_seconds{{pid=\"{pid}\",quantile=\"0.5\"}} {:.6}\n",
                h.p50_ns as f64 / 1e9
            ));
            out.push_str(&format!(
                "zernel_cuda_launch_latency_seconds{{pid=\"{pid}\",quantile=\"0.99\"}} {:.6}\n",
                h.p99_ns as f64 / 1e9
            ));
            out.push_str(&format!(
                "zernel_cuda_launch_latency_seconds_count{{pid=\"{pid}\"}} {}\n",
                h.count
            ));
        }

        // NCCL
        out.push_str("# HELP zernel_nccl_collective_duration_seconds NCCL collective duration\n");
        out.push_str("# TYPE zernel_nccl_collective_duration_seconds summary\n");
        for (op, h) in &self.nccl_duration {
            out.push_str(&format!(
                "zernel_nccl_collective_duration_seconds{{op=\"{op}\",quantile=\"0.5\"}} {:.6}\n",
                h.p50_ns as f64 / 1e9
            ));
            out.push_str(&format!(
                "zernel_nccl_collective_duration_seconds{{op=\"{op}\",quantile=\"0.99\"}} {:.6}\n",
                h.p99_ns as f64 / 1e9
            ));
        }

        // DataLoader
        out.push_str("# HELP zernel_dataloader_wait_seconds DataLoader wait time\n");
        out.push_str("# TYPE zernel_dataloader_wait_seconds summary\n");
        for (pid, h) in &self.dataloader_wait {
            out.push_str(&format!(
                "zernel_dataloader_wait_seconds{{pid=\"{pid}\",quantile=\"0.5\"}} {:.6}\n",
                h.p50_ns as f64 / 1e9
            ));
            out.push_str(&format!(
                "zernel_dataloader_wait_seconds{{pid=\"{pid}\",quantile=\"0.99\"}} {:.6}\n",
                h.p99_ns as f64 / 1e9
            ));
        }

        out
    }

    /// Build a JSON snapshot for WebSocket push to CLI.
    pub fn to_ws_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "gpu_utilization": self.gpu_memory.iter().map(|(k, v)| {
                serde_json::json!({
                    "key": k,
                    "current_bytes": v.current_bytes,
                    "peak_bytes": v.peak_bytes,
                })
            }).collect::<Vec<_>>(),
            "cuda_latency_p50_us": self.cuda_latency.values().next()
                .map(|h| h.p50_ns as f64 / 1000.0).unwrap_or(0.0),
            "cuda_latency_p99_us": self.cuda_latency.values().next()
                .map(|h| h.p99_ns as f64 / 1000.0).unwrap_or(0.0),
            "nccl_allreduce_p50_ms": self.nccl_duration.get("all_reduce")
                .map(|h| h.p50_ns as f64 / 1e6).unwrap_or(0.0),
            "nccl_allreduce_p99_ms": self.nccl_duration.get("all_reduce")
                .map(|h| h.p99_ns as f64 / 1e6).unwrap_or(0.0),
            "dataloader_wait_p50_ms": self.dataloader_wait.values().next()
                .map(|h| h.p50_ns as f64 / 1e6).unwrap_or(0.0),
            "last_update_ms": self.last_update_ms,
        })
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct GpuMemMetrics {
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub alloc_count: u64,
    pub free_count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyHistogram {
    pub count: u64,
    pub sum_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub p50_ns: u64,
    pub p99_ns: u64,
    #[serde(skip)]
    samples: Vec<u64>,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self {
            count: 0,
            sum_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
            p50_ns: 0,
            p99_ns: 0,
            samples: Vec::new(),
        }
    }
}

impl LatencyHistogram {
    pub fn record(&mut self, value_ns: u64) {
        self.count += 1;
        self.sum_ns += value_ns;
        if value_ns < self.min_ns {
            self.min_ns = value_ns;
        }
        if value_ns > self.max_ns {
            self.max_ns = value_ns;
        }
        self.samples.push(value_ns);

        // Recompute percentiles every 100 samples or on first 10
        if self.samples.len() <= 10 || self.samples.len().is_multiple_of(100) {
            self.recompute_percentiles();
        }
    }

    fn recompute_percentiles(&mut self) {
        if self.samples.is_empty() {
            return;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let len = sorted.len();
        self.p50_ns = sorted[len / 2];
        self.p99_ns = sorted[((len as f64 * 0.99) as usize).min(len - 1)];
    }

    pub fn mean_ns(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_ns as f64 / self.count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_percentiles() {
        let mut h = LatencyHistogram::default();
        for i in 1..=100 {
            h.record(i * 1000);
        }
        assert_eq!(h.count, 100);
        assert_eq!(h.min_ns, 1000);
        assert_eq!(h.max_ns, 100_000);
        assert!(h.p50_ns > 0);
        assert!(h.p99_ns >= h.p50_ns);
    }

    #[test]
    fn prometheus_format() {
        let mut m = AggregatedMetrics::default();
        m.record_gpu_mem(1234, 0, 84934656, 84934656);
        m.record_cuda_latency(1234, 142000);
        let prom = m.to_prometheus();
        assert!(prom.contains("zernel_gpu_memory_used_bytes"));
        assert!(prom.contains("84934656"));
    }

    #[test]
    fn ws_snapshot_is_valid_json() {
        let m = AggregatedMetrics::default();
        let snap = m.to_ws_snapshot();
        assert!(snap.is_object());
    }
}
