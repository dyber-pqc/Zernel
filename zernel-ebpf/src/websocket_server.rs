// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::aggregation::AggregatedMetrics;
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, error, info, warn};

/// WebSocket server for real-time telemetry streaming to `zernel watch`.
pub struct WebSocketServer {
    metrics: Arc<RwLock<AggregatedMetrics>>,
    port: u16,
    push_interval_ms: u64,
}

impl WebSocketServer {
    pub fn new(metrics: Arc<RwLock<AggregatedMetrics>>, port: u16, push_interval_ms: u64) -> Self {
        Self {
            metrics,
            port,
            push_interval_ms,
        }
    }

    pub async fn serve(&self) -> Result<()> {
        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));
        let listener = TcpListener::bind(addr).await?;
        info!(port = self.port, "WebSocket server listening");

        let metrics = Arc::clone(&self.metrics);
        let interval_ms = self.push_interval_ms;

        loop {
            let (stream, peer) = listener.accept().await?;
            let metrics = Arc::clone(&metrics);
            info!(?peer, "WebSocket client connected");

            tokio::spawn(async move {
                match accept_async(stream).await {
                    Ok(ws) => {
                        if let Err(e) = handle_connection(ws, metrics, interval_ms).await {
                            debug!(?peer, error = %e, "WebSocket connection ended");
                        }
                    }
                    Err(e) => {
                        warn!(?peer, error = %e, "WebSocket handshake failed");
                    }
                }
            });
        }
    }
}

async fn handle_connection(
    ws: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    metrics: Arc<RwLock<AggregatedMetrics>>,
    interval_ms: u64,
) -> Result<()> {
    let (mut write, mut read) = ws.split();
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let snapshot = {
                    let m = metrics.read().await;
                    m.to_ws_snapshot()
                };
                let msg = Message::Text(snapshot.to_string());
                if write.send(msg).await.is_err() {
                    break;
                }
            }
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(data))) => {
                        let _ = write.send(Message::Pong(data)).await;
                    }
                    Some(Err(e)) => {
                        error!("WebSocket read error: {e}");
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}
