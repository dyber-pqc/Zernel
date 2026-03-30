// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! HTTP routes for the Zernel web dashboard.

use crate::state::AppState;
use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        Html,
    },
    routing::get,
    Router,
};
use futures_util::stream::Stream;
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

/// Build the complete router with all dashboard routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/experiments", get(experiments))
        .route("/jobs", get(jobs))
        .route("/models", get(models))
        .route("/api/sse", get(sse_handler))
        .with_state(state)
}

// ============================================================
// Page handlers
// ============================================================

async fn index() -> Html<String> {
    Html(page(
        "Overview",
        r#"
        <h2>GPU Telemetry</h2>
        <div id="gpu-section" hx-ext="sse" sse-connect="/api/sse" sse-swap="telemetry">
            <p class="muted">Connecting to zerneld...</p>
        </div>

        <h2>eBPF Metrics</h2>
        <div id="telemetry-section">
            <p class="muted">Waiting for data...</p>
        </div>
        "#,
    ))
}

async fn experiments(State(state): State<Arc<AppState>>) -> Html<String> {
    let rows = if state.experiments_db.exists() {
        match rusqlite::Connection::open(&state.experiments_db) {
            Ok(conn) => {
                let mut stmt = conn
                    .prepare("SELECT id, name, status, metrics, created_at, duration_secs FROM experiments ORDER BY created_at DESC LIMIT 50")
                    .unwrap_or_else(|_| conn.prepare("SELECT 1").unwrap());

                let mut rows_html = String::new();
                let mut query_rows = stmt.query([]).unwrap();
                while let Ok(Some(row)) = query_rows.next() {
                    let id: String = row.get(0).unwrap_or_default();
                    let name: String = row.get(1).unwrap_or_default();
                    let status: String = row.get(2).unwrap_or_default();
                    let metrics: String = row.get(3).unwrap_or_default();
                    let created: String = row.get(4).unwrap_or_default();
                    let duration: Option<f64> = row.get(5).unwrap_or(None);

                    let dur_str = duration
                        .map(|d| format!("{d:.1}s"))
                        .unwrap_or_else(|| "-".into());
                    let loss = serde_json::from_str::<serde_json::Value>(&metrics)
                        .ok()
                        .and_then(|m| m["loss"].as_f64())
                        .map(|v| format!("{v:.4}"))
                        .unwrap_or_else(|| "-".into());

                    let status_class = match status.trim_matches('"') {
                        "\"Done\"" | "Done" => "status-done",
                        "\"Failed\"" | "Failed" => "status-failed",
                        "\"Running\"" | "Running" => "status-running",
                        _ => "",
                    };

                    rows_html.push_str(&format!(
                        "<tr><td>{id}</td><td>{name}</td><td class=\"{status_class}\">{status}</td><td>{loss}</td><td>{dur_str}</td><td>{}</td></tr>",
                        &created[..19.min(created.len())]
                    ));
                }
                rows_html
            }
            Err(_) => "<tr><td colspan='6'>Could not open database</td></tr>".into(),
        }
    } else {
        "<tr><td colspan='6'>No experiments yet. Run: zernel run &lt;script&gt;</td></tr>".into()
    };

    Html(page(
        "Experiments",
        &format!(
            r#"
        <h2>Experiments</h2>
        <table class="data-table">
            <thead><tr><th>ID</th><th>Name</th><th>Status</th><th>Loss</th><th>Duration</th><th>Created</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        "#
        ),
    ))
}

async fn jobs(State(state): State<Arc<AppState>>) -> Html<String> {
    let rows = if state.jobs_db.exists() {
        match rusqlite::Connection::open(&state.jobs_db) {
            Ok(conn) => {
                let mut rows_html = String::new();
                if let Ok(mut stmt) = conn.prepare("SELECT id, script, status, gpus_per_node, nodes, framework, exit_code FROM jobs ORDER BY submitted_at DESC LIMIT 50") {
                    let mut query_rows = stmt.query([]).unwrap();
                    while let Ok(Some(row)) = query_rows.next() {
                        let id: String = row.get(0).unwrap_or_default();
                        let script: String = row.get(1).unwrap_or_default();
                        let status: String = row.get(2).unwrap_or_default();
                        let gpus: u32 = row.get(3).unwrap_or(0);
                        let nodes: u32 = row.get(4).unwrap_or(0);
                        let fw: String = row.get(5).unwrap_or_default();
                        let exit: Option<i32> = row.get(6).unwrap_or(None);
                        let exit_str = exit.map(|e| e.to_string()).unwrap_or_else(|| "-".into());

                        rows_html.push_str(&format!(
                            "<tr><td>{id}</td><td>{script}</td><td>{status}</td><td>{gpus}</td><td>{nodes}</td><td>{fw}</td><td>{exit_str}</td></tr>"
                        ));
                    }
                }
                if rows_html.is_empty() {
                    "<tr><td colspan='7'>No jobs yet</td></tr>".into()
                } else {
                    rows_html
                }
            }
            Err(_) => "<tr><td colspan='7'>Could not open database</td></tr>".into(),
        }
    } else {
        "<tr><td colspan='7'>No jobs yet. Run: zernel job submit &lt;script&gt;</td></tr>".into()
    };

    Html(page(
        "Jobs",
        &format!(
            r#"
        <h2>Jobs</h2>
        <table class="data-table">
            <thead><tr><th>ID</th><th>Script</th><th>Status</th><th>GPUs</th><th>Nodes</th><th>Framework</th><th>Exit</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        "#
        ),
    ))
}

async fn models(State(state): State<Arc<AppState>>) -> Html<String> {
    let rows = if state.models_registry.exists() {
        match std::fs::read_to_string(&state.models_registry) {
            Ok(data) => {
                let entries: Vec<serde_json::Value> =
                    serde_json::from_str(&data).unwrap_or_default();
                let mut rows_html = String::new();
                for entry in &entries {
                    let name = entry["name"].as_str().unwrap_or("-");
                    let version = entry["version"].as_str().unwrap_or("-");
                    let tag = entry["tag"].as_str().unwrap_or("-");
                    let size = entry["size_bytes"].as_u64().unwrap_or(0);
                    let saved = entry["saved_at"].as_str().unwrap_or("-");
                    let size_str = format!("{:.1} MB", size as f64 / (1024.0 * 1024.0));

                    rows_html.push_str(&format!(
                        "<tr><td>{name}</td><td>{version}</td><td>{tag}</td><td>{size_str}</td><td>{}</td></tr>",
                        &saved[..10.min(saved.len())]
                    ));
                }
                if rows_html.is_empty() {
                    "<tr><td colspan='5'>No models yet</td></tr>".into()
                } else {
                    rows_html
                }
            }
            Err(_) => "<tr><td colspan='5'>Could not read registry</td></tr>".into(),
        }
    } else {
        "<tr><td colspan='5'>No models yet. Run: zernel model save &lt;path&gt;</td></tr>".into()
    };

    Html(page(
        "Models",
        &format!(
            r#"
        <h2>Models</h2>
        <table class="data-table">
            <thead><tr><th>Name</th><th>Version</th><th>Tag</th><th>Size</th><th>Saved</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        "#
        ),
    ))
}

// ============================================================
// SSE handler
// ============================================================

async fn sse_handler(
    State(state): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.sse_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|msg| match msg {
        Ok(html) => Some(Ok(Event::default().event("telemetry").data(html))),
        Err(_) => None,
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ============================================================
// Page template
// ============================================================

fn page(title: &str, content: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Zernel — {title}</title>
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <script src="https://unpkg.com/htmx-ext-sse@2.2.2/sse.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }}
        nav {{ background: #1e293b; padding: 12px 24px; display: flex; gap: 24px; align-items: center; border-bottom: 1px solid #334155; }}
        nav a {{ color: #94a3b8; text-decoration: none; font-size: 14px; }} nav a:hover {{ color: #f1f5f9; }}
        nav .brand {{ color: #22d3ee; font-weight: 700; font-size: 18px; margin-right: 16px; }}
        main {{ max-width: 1200px; margin: 24px auto; padding: 0 24px; }}
        h2 {{ color: #f1f5f9; margin: 24px 0 12px; font-size: 20px; }}
        .muted {{ color: #64748b; }}
        .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .data-table th {{ text-align: left; padding: 8px 12px; background: #1e293b; color: #94a3b8; border-bottom: 1px solid #334155; }}
        .data-table td {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; }}
        .data-table tr:hover {{ background: #1e293b; }}
        .status-done {{ color: #22c55e; }} .status-failed {{ color: #ef4444; }} .status-running {{ color: #eab308; }}
        .gpu-card {{ display: inline-block; width: 280px; margin: 8px; padding: 12px; background: #1e293b; border-radius: 8px; }}
        .gpu-label {{ font-size: 13px; color: #94a3b8; margin-bottom: 6px; }}
        .gpu-bar-bg {{ height: 20px; background: #334155; border-radius: 4px; overflow: hidden; }}
        .gpu-bar {{ height: 100%; border-radius: 4px; transition: width 0.5s; }}
        .gpu-pct {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
        table {{ border-collapse: collapse; }} table td {{ padding: 4px 16px 4px 0; font-size: 14px; }}
        footer {{ text-align: center; color: #475569; font-size: 12px; margin-top: 48px; padding: 24px; }}
    </style>
</head>
<body>
    <nav>
        <span class="brand">Zernel</span>
        <a href="/">Overview</a>
        <a href="/experiments">Experiments</a>
        <a href="/jobs">Jobs</a>
        <a href="/models">Models</a>
    </nav>
    <main>
        {content}
    </main>
    <footer>Zernel Dashboard &mdash; Copyright &copy; 2026 Dyber, Inc.</footer>
</body>
</html>"#
    )
}
