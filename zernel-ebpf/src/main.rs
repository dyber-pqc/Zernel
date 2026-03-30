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
use clap::Parser;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

#[derive(Parser)]
#[command(name = "zerneld")]
#[command(about = "Zernel eBPF Observability Daemon")]
#[command(version)]
struct Args {
    /// Prometheus metrics HTTP port
    #[arg(long, default_value = "9091", env = "ZERNEL_METRICS_PORT")]
    metrics_port: u16,

    /// WebSocket telemetry stream port
    #[arg(long, default_value = "9092", env = "ZERNEL_WS_PORT")]
    ws_port: u16,

    /// WebSocket push interval in milliseconds
    #[arg(long, default_value = "1000", env = "ZERNEL_PUSH_INTERVAL_MS")]
    push_interval_ms: u64,

    /// Run with simulated telemetry (no BPF probes)
    #[arg(long)]
    simulate: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("ZERNEL_LOG").unwrap_or_else(|_| "zernel_ebpf=info".into()))
        .init();

    let args = Args::parse();

    info!("zerneld v{}", env!("CARGO_PKG_VERSION"));

    // Load BPF probes (or stub mode)
    let bpf_active = loader::load_all_probes().is_ok();

    // Shared metrics state
    let metrics = Arc::new(RwLock::new(AggregatedMetrics::default()));

    // If no BPF probes or --simulate, run simulator
    let simulate = args.simulate || !bpf_active;
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
            for gpu in m.gpu_memory.values() {
                if gpu.peak_bytes > 0 {
                    let used_pct = gpu.current_bytes as f64 / gpu.peak_bytes as f64 * 100.0;
                    alert_engine.evaluate("gpu_memory_used_pct", used_pct);
                }
            }
        }
    });

    // Start servers
    let metrics_srv = metrics_server::MetricsServer::new(Arc::clone(&metrics), args.metrics_port);
    let ws_srv = websocket_server::WebSocketServer::new(
        Arc::clone(&metrics),
        args.ws_port,
        args.push_interval_ms,
    );

    info!(
        metrics_port = args.metrics_port,
        ws_port = args.ws_port,
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
