// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::aggregation::AggregatedMetrics;
use anyhow::Result;
use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::{error, info};

/// Prometheus-compatible metrics server.
pub struct MetricsServer {
    metrics: Arc<RwLock<AggregatedMetrics>>,
    port: u16,
}

impl MetricsServer {
    pub fn new(metrics: Arc<RwLock<AggregatedMetrics>>, port: u16) -> Self {
        Self { metrics, port }
    }

    pub async fn serve(&self) -> Result<()> {
        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));
        let listener = TcpListener::bind(addr).await?;
        info!(port = self.port, "Prometheus metrics server listening");

        let metrics = Arc::clone(&self.metrics);

        loop {
            let (stream, _) = listener.accept().await?;
            let io = TokioIo::new(stream);
            let metrics = Arc::clone(&metrics);

            tokio::spawn(async move {
                let service = service_fn(move |req: Request<hyper::body::Incoming>| {
                    let metrics = Arc::clone(&metrics);
                    async move { handle_request(req, metrics).await }
                });

                if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                    error!("HTTP error: {err}");
                }
            });
        }
    }
}

async fn handle_request(
    req: Request<hyper::body::Incoming>,
    metrics: Arc<RwLock<AggregatedMetrics>>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    match req.uri().path() {
        "/metrics" => {
            let m = metrics.read().await;
            let body = m.to_prometheus();
            Ok(Response::builder()
                .header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                .body(Full::new(Bytes::from(body)))
                .unwrap())
        }
        "/health" => Ok(Response::new(Full::new(Bytes::from("ok")))),
        "/json" => {
            let m = metrics.read().await;
            let body = serde_json::to_string_pretty(&*m).unwrap_or_default();
            Ok(Response::builder()
                .header("Content-Type", "application/json")
                .body(Full::new(Bytes::from(body)))
                .unwrap())
        }
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from("not found")))
            .unwrap()),
    }
}
