// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::Result;
use serde::Deserialize;
use tracing::debug;

/// Telemetry snapshot received from zerneld via WebSocket.
#[derive(Debug, Deserialize)]
pub struct TelemetrySnapshot {
    pub gpu_utilization: Vec<GpuStatus>,
    pub cuda_latency_p50_us: f64,
    pub cuda_latency_p99_us: f64,
    pub nccl_allreduce_p50_ms: f64,
    pub nccl_allreduce_p99_ms: f64,
    pub dataloader_wait_p50_ms: f64,
    pub pcie_bandwidth_gbps: f64,
    pub scheduler_phase: String,
}

#[derive(Debug, Deserialize)]
pub struct GpuStatus {
    pub gpu_id: u32,
    pub utilization_pct: u8,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
}

/// Client that connects to zerneld WebSocket and receives telemetry.
pub struct TelemetryClient {
    url: String,
}

impl TelemetryClient {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            url: format!("ws://{host}:{port}"),
        }
    }

    /// Connect to zerneld and start receiving snapshots.
    pub async fn connect(&self) -> Result<()> {
        debug!(url = self.url, "connecting to zerneld");
        // TODO: tokio-tungstenite WebSocket connection
        // Parse incoming JSON messages as TelemetrySnapshot
        Ok(())
    }
}
