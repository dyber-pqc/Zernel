// Copyright (C) 2026 Dyber, Inc. — Proprietary
//
// Zernel Web Dashboard — browser-based ML monitoring
//
// Connects to zerneld for real-time GPU telemetry, reads experiment/job/model
// data from SQLite, serves an htmx-powered dashboard with Server-Sent Events.

#![allow(dead_code)]

mod routes;
mod sse;
mod state;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::info;

#[derive(Parser)]
#[command(name = "zernel-dashboard")]
#[command(about = "Zernel Web Dashboard — browser-based ML monitoring")]
#[command(version)]
struct Args {
    /// HTTP port to serve the dashboard
    #[arg(long, default_value = "3000", env = "ZERNEL_DASHBOARD_PORT")]
    port: u16,

    /// zerneld WebSocket URL for real-time telemetry
    #[arg(long, default_value = "ws://127.0.0.1:9092", env = "ZERNEL_WS_URL")]
    zerneld_url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("ZERNEL_LOG").unwrap_or_else(|_| "zernel_dashboard=info".into()),
        )
        .init();

    let args = Args::parse();

    info!("Zernel Dashboard v{}", env!("CARGO_PKG_VERSION"));

    // Broadcast channel for SSE (telemetry updates)
    let (sse_tx, _) = broadcast::channel::<String>(64);

    // Shared application state
    let app_state = Arc::new(state::AppState::new(sse_tx.clone()));

    // Start zerneld relay in background
    let relay_url = args.zerneld_url.clone();
    let relay_tx = sse_tx.clone();
    tokio::spawn(async move {
        sse::relay_zerneld_to_sse(&relay_url, relay_tx).await;
    });

    // Build router
    let app = routes::build_router(app_state);

    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!(
        port = args.port,
        zerneld = args.zerneld_url,
        "dashboard ready at http://localhost:{}",
        args.port
    );

    axum::serve(listener, app).await?;

    Ok(())
}
