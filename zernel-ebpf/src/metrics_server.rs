// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::aggregation::AggregatedMetrics;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Prometheus-compatible metrics server.
///
/// Exposes metrics on port 9091 at /metrics in Prometheus text exposition format.
pub struct MetricsServer {
    metrics: Arc<RwLock<AggregatedMetrics>>,
    port: u16,
}

impl MetricsServer {
    pub fn new(metrics: Arc<RwLock<AggregatedMetrics>>, port: u16) -> Self {
        Self { metrics, port }
    }

    /// Start serving metrics. Runs until cancelled.
    pub async fn serve(&self) -> Result<()> {
        info!(port = self.port, "metrics server starting");

        // TODO: Implement HTTP server with Prometheus text format
        // Using warp or axum, expose /metrics endpoint that reads
        // from self.metrics and formats as:
        //
        // zernel_gpu_memory_used_bytes{pid="1234",gpu_id="0"} 84934656
        // zernel_cuda_launch_latency_seconds{pid="1234",quantile="0.5"} 0.000142
        // zernel_nccl_collective_duration_seconds{op="all_reduce",quantile="0.99"} 0.067
        // etc.

        info!(port = self.port, "metrics server ready");
        Ok(())
    }
}
