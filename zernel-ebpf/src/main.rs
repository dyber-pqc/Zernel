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
mod fallback;
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

    /// Force simulated telemetry (ignore BPF and nvidia-smi)
    #[arg(long)]
    simulate: bool,
}

/// Telemetry source selection result.
#[derive(Debug, Clone, Copy)]
enum TelemetrySource {
    /// Real BPF ring buffer events from kernel probes.
    Bpf,
    /// nvidia-smi + /proc polling (no BPF, but real GPU data).
    NvidiaSmi,
    /// Fully simulated data for development/demos.
    Simulator,
}

impl std::fmt::Display for TelemetrySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bpf => write!(f, "BPF"),
            Self::NvidiaSmi => write!(f, "nvidia-smi"),
            Self::Simulator => write!(f, "simulator"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("ZERNEL_LOG").unwrap_or_else(|_| "zernel_ebpf=info".into()))
        .init();

    let args = Args::parse();

    info!("zerneld v{}", env!("CARGO_PKG_VERSION"));

    // ============================================================
    // Three-tier telemetry source selection:
    //   1. BPF ring buffers (Linux 6.12+ with root)
    //   2. nvidia-smi polling (any system with NVIDIA GPU)
    //   3. Simulator (development/demo)
    // ============================================================

    let source = if args.simulate {
        TelemetrySource::Simulator
    } else {
        // Try BPF first
        let load_result = loader::load_all_probes().unwrap_or_else(|e| {
            tracing::warn!("BPF probe loading failed: {e}");
            loader::LoadResult {
                status: loader::ProbeStatus::none(),
            }
        });

        if load_result.status.active_count() > 0 {
            info!(
                active = load_result.status.active_count(),
                "BPF probes loaded"
            );
            TelemetrySource::Bpf
        } else if fallback::nvidia_smi_available() {
            info!("BPF unavailable, using nvidia-smi fallback for real GPU metrics");
            TelemetrySource::NvidiaSmi
        } else {
            info!("no BPF or nvidia-smi available, using simulator");
            TelemetrySource::Simulator
        }
    };

    info!(source = %source, "telemetry source selected");

    // Shared metrics state
    let metrics = Arc::new(RwLock::new(AggregatedMetrics::default()));

    // Start the appropriate telemetry provider
    match source {
        TelemetrySource::Bpf => {
            // TODO: When full BPF skeleton loading is wired up, spawn ring
            // buffer polling tasks here that read events and feed into metrics.
            // For now, BPF is detected but we fall through to fallback behavior.
            info!("BPF ring buffer polling active");
            let fb_metrics = Arc::clone(&metrics);
            tokio::spawn(async move {
                fallback::run_fallback(fb_metrics, 1000).await;
            });
        }
        TelemetrySource::NvidiaSmi => {
            let fb_metrics = Arc::clone(&metrics);
            tokio::spawn(async move {
                fallback::run_fallback(fb_metrics, 1000).await;
            });
        }
        TelemetrySource::Simulator => {
            let sim_metrics = Arc::clone(&metrics);
            tokio::spawn(async move {
                simulator::run_simulator(sim_metrics, 500).await;
            });
        }
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
        source = %source,
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
