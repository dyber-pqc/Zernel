// Copyright (C) 2026 Dyber, Inc. — Proprietary

use std::path::PathBuf;
use tokio::sync::broadcast;

/// Shared application state for the dashboard.
pub struct AppState {
    /// Broadcast channel for SSE telemetry updates (HTML fragments).
    pub sse_tx: broadcast::Sender<String>,
    /// Path to the experiments SQLite database.
    pub experiments_db: PathBuf,
    /// Path to the jobs SQLite database.
    pub jobs_db: PathBuf,
    /// Path to the model registry JSON.
    pub models_registry: PathBuf,
}

impl AppState {
    pub fn new(sse_tx: broadcast::Sender<String>) -> Self {
        let zernel_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".zernel");

        Self {
            sse_tx,
            experiments_db: zernel_dir.join("experiments").join("experiments.db"),
            jobs_db: zernel_dir.join("jobs").join("jobs.db"),
            models_registry: zernel_dir.join("models").join("registry.json"),
        }
    }
}
