// Copyright (C) 2026 Dyber, Inc. — GPL-2.0
//
// zerneld — Zernel eBPF Observability Daemon
//
// Allow dead code: probe consumers and event types are public API
// surface consumed when BPF probes are wired up on Linux.
#![allow(dead_code)]

// Loads eBPF probes, consumes events from ring buffers, aggregates metrics,
// and exposes them via Prometheus endpoint and WebSocket for the CLI IDE.

mod aggregation;
mod alerts;
mod consumers;
mod loader;
mod metrics_server;
mod simulator;
mod websocket_server;

use aggregation::AggregatedMetrics;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

const METRICS_PORT: u16 = 9091;
const WS_PORT: u16 = 9092;
const PUSH_INTERVAL_MS: u64 = 1000;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("ZERNEL_LOG").unwrap_or_else(|_| "zernel_ebpf=info".into()))
        .init();

    info!("zerneld v{}", env!("CARGO_PKG_VERSION"));

    // Load BPF probes (or stub mode)
    let bpf_active = loader::load_all_probes().is_ok();

    // Shared metrics state
    let metrics = Arc::new(RwLock::new(AggregatedMetrics::default()));

    // If no BPF probes, run simulator for development
    let simulate = std::env::args().any(|a| a == "--simulate") || !bpf_active;
    if simulate {
        info!("running telemetry simulator (no BPF probes)");
        let sim_metrics = Arc::clone(&metrics);
        tokio::spawn(async move {
            simulator::run_simulator(sim_metrics, 500).await;
        });
    }

    // Alert engine
    let alert_metrics = Arc::clone(&metrics);
    let alert_engine = alerts::AlertEngine::new(vec![alerts::AlertRule {
        name: "gpu_oom_warning".into(),
        metric: "gpu_memory_used_pct".into(),
        threshold: 95.0,
        comparison: alerts::Comparison::GreaterThan,
        action: alerts::AlertAction::Log,
    }]);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            let m = alert_metrics.read().await;
            for (key, gpu) in &m.gpu_memory {
                if gpu.peak_bytes > 0 {
                    let used_pct = gpu.current_bytes as f64 / gpu.peak_bytes as f64 * 100.0;
                    alert_engine.evaluate("gpu_memory_used_pct", used_pct);
                }
                // Suppress unused key warning
                let _ = key;
            }
        }
    });

    // Start servers
    let metrics_srv = metrics_server::MetricsServer::new(Arc::clone(&metrics), METRICS_PORT);
    let ws_srv =
        websocket_server::WebSocketServer::new(Arc::clone(&metrics), WS_PORT, PUSH_INTERVAL_MS);

    info!(
        metrics_port = METRICS_PORT,
        ws_port = WS_PORT,
        simulate,
        "zerneld ready"
    );

    tokio::select! {
        res = metrics_srv.serve() => {
            if let Err(e) = res {
                tracing::error!("metrics server error: {e}");
            }
        }
        res = ws_srv.serve() => {
            if let Err(e) = res {
                tracing::error!("WebSocket server error: {e}");
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("shutting down");
        }
    }

    Ok(())
}
