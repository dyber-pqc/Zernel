// Copyright (C) 2026 Dyber, Inc. — GPL-2.0
//
// zerneld — Zernel eBPF Observability Daemon
//
// Loads eBPF probes, consumes events from ring buffers, aggregates metrics,
// and exposes them via Prometheus endpoint and WebSocket for the CLI IDE.

mod aggregation;
mod alerts;
mod consumers;
mod loader;
mod metrics_server;
mod websocket_server;

use aggregation::AggregatedMetrics;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("zernel_ebpf=info")
        .init();

    info!("zerneld starting");

    // Load BPF probes
    loader::load_all_probes()?;

    // Shared metrics state
    let metrics = Arc::new(RwLock::new(AggregatedMetrics::default()));

    // Start Prometheus metrics server
    let metrics_srv = metrics_server::MetricsServer::new(Arc::clone(&metrics), 9091);

    // Start WebSocket server for CLI IDE
    let ws_srv = websocket_server::WebSocketServer::new(Arc::clone(&metrics), 9092, 1000);

    info!("zerneld ready");

    // Run servers concurrently
    tokio::select! {
        res = metrics_srv.serve() => { res?; }
        res = ws_srv.serve() => { res?; }
        _ = tokio::signal::ctrl_c() => {
            info!("shutting down");
        }
    }

    Ok(())
}
