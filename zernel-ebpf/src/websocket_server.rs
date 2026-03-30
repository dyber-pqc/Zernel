// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::aggregation::AggregatedMetrics;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// WebSocket server for real-time telemetry streaming to `zernel watch`.
///
/// Pushes metric snapshots to connected CLI clients at a configurable interval.
pub struct WebSocketServer {
    metrics: Arc<RwLock<AggregatedMetrics>>,
    port: u16,
    push_interval_ms: u64,
}

impl WebSocketServer {
    pub fn new(
        metrics: Arc<RwLock<AggregatedMetrics>>,
        port: u16,
        push_interval_ms: u64,
    ) -> Self {
        Self {
            metrics,
            port,
            push_interval_ms,
        }
    }

    pub async fn serve(&self) -> Result<()> {
        info!(
            port = self.port,
            interval_ms = self.push_interval_ms,
            "WebSocket server starting"
        );

        // TODO: Accept WebSocket connections and push JSON metric snapshots
        // at self.push_interval_ms interval.

        Ok(())
    }
}
