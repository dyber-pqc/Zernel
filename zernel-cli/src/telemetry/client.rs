// Copyright (C) 2026 Dyber, Inc. — Proprietary

use futures_util::StreamExt;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tracing::{debug, warn};

/// Telemetry snapshot received from zerneld via WebSocket.
/// Must match the JSON format from AggregatedMetrics::to_ws_snapshot().
#[derive(Debug, Clone, Deserialize)]
pub struct TelemetrySnapshot {
    #[serde(default)]
    pub gpu_utilization: Vec<GpuEntry>,
    #[serde(default)]
    pub cuda_latency_p50_us: f64,
    #[serde(default)]
    pub cuda_latency_p99_us: f64,
    #[serde(default)]
    pub nccl_allreduce_p50_ms: f64,
    #[serde(default)]
    pub nccl_allreduce_p99_ms: f64,
    #[serde(default)]
    pub dataloader_wait_p50_ms: f64,
    #[serde(default)]
    pub last_update_ms: u64,
}

/// GPU memory entry from zerneld (matches to_ws_snapshot format).
#[derive(Debug, Clone, Deserialize)]
pub struct GpuEntry {
    pub key: String,
    pub current_bytes: u64,
    pub peak_bytes: u64,
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

    /// Try to connect to zerneld. Returns a receiver channel if successful.
    /// The connection runs in a background task with automatic reconnection.
    pub async fn try_connect(&self) -> Option<mpsc::UnboundedReceiver<TelemetrySnapshot>> {
        let url = self.url.clone();

        // Quick connectivity check
        match connect_async(&url).await {
            Ok((ws, _)) => {
                let (tx, rx) = mpsc::unbounded_channel();
                let url_clone = url.clone();

                tokio::spawn(async move {
                    Self::reader_loop(ws, tx.clone(), url_clone).await;
                });

                debug!(url = self.url, "connected to zerneld");
                Some(rx)
            }
            Err(e) => {
                debug!(url = self.url, error = %e, "could not connect to zerneld");
                None
            }
        }
    }

    async fn reader_loop(
        ws: tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        tx: mpsc::UnboundedSender<TelemetrySnapshot>,
        url: String,
    ) {
        let (_write, mut read) = ws.split();
        let mut backoff_ms = 1000u64;

        loop {
            match read.next().await {
                Some(Ok(msg)) => {
                    if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                        match serde_json::from_str::<TelemetrySnapshot>(&text) {
                            Ok(snapshot) => {
                                if tx.send(snapshot).is_err() {
                                    // Receiver dropped
                                    return;
                                }
                                backoff_ms = 1000; // reset on success
                            }
                            Err(e) => {
                                debug!(error = %e, "failed to parse telemetry snapshot");
                            }
                        }
                    }
                }
                Some(Err(e)) => {
                    warn!(error = %e, "WebSocket error, reconnecting...");
                    break;
                }
                None => {
                    warn!("WebSocket closed, reconnecting...");
                    break;
                }
            }
        }

        // Reconnection with exponential backoff
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
            match connect_async(&url).await {
                Ok((ws, _)) => {
                    debug!("reconnected to zerneld");
                    // Recurse into reader loop with new connection
                    Box::pin(Self::reader_loop(ws, tx, url)).await;
                    return;
                }
                Err(e) => {
                    debug!(error = %e, backoff_ms, "reconnect failed");
                    backoff_ms = (backoff_ms * 2).min(30_000);
                }
            }
        }
    }
}

/// Get the zerneld WebSocket port from environment or default.
pub fn ws_port() -> u16 {
    std::env::var("ZERNEL_WS_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9092)
}

/// Get the zerneld metrics port from environment or default.
pub fn metrics_port() -> u16 {
    std::env::var("ZERNEL_METRICS_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9091)
}
