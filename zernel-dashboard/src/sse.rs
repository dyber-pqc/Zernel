// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! Relays zerneld WebSocket telemetry to SSE broadcast for the dashboard.

use futures_util::StreamExt;
use tokio::sync::broadcast;
use tokio_tungstenite::connect_async;
use tracing::{debug, info, warn};

/// Connect to zerneld WebSocket and relay telemetry as SSE-formatted HTML.
pub async fn relay_zerneld_to_sse(url: &str, tx: broadcast::Sender<String>) {
    let mut backoff_ms = 1000u64;

    loop {
        info!(url, "connecting to zerneld...");

        match connect_async(url).await {
            Ok((ws, _)) => {
                info!("connected to zerneld");
                backoff_ms = 1000;

                let (_write, mut read) = ws.split();

                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                                let html = render_telemetry_html(&data);
                                let _ = tx.send(html);
                            }
                        }
                        Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => {
                            info!("zerneld WebSocket closed");
                            break;
                        }
                        Err(e) => {
                            warn!(error = %e, "WebSocket error");
                            break;
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                debug!(error = %e, backoff_ms, "failed to connect to zerneld");
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
        backoff_ms = (backoff_ms * 2).min(30_000);
    }
}

/// Render telemetry JSON as an HTML fragment for htmx SSE swap.
fn render_telemetry_html(data: &serde_json::Value) -> String {
    let cuda_p50 = data["cuda_latency_p50_us"].as_f64().unwrap_or(0.0);
    let cuda_p99 = data["cuda_latency_p99_us"].as_f64().unwrap_or(0.0);
    let nccl_p50 = data["nccl_allreduce_p50_ms"].as_f64().unwrap_or(0.0);
    let nccl_p99 = data["nccl_allreduce_p99_ms"].as_f64().unwrap_or(0.0);
    let dl_p50 = data["dataloader_wait_p50_ms"].as_f64().unwrap_or(0.0);

    let mut gpu_html = String::new();
    if let Some(gpus) = data["gpu_utilization"].as_array() {
        for gpu in gpus {
            let key = gpu["key"].as_str().unwrap_or("?");
            let used = gpu["current_bytes"].as_u64().unwrap_or(0);
            let peak = gpu["peak_bytes"].as_u64().unwrap_or(1);
            let pct = if peak > 0 {
                (used as f64 / peak as f64 * 100.0) as u32
            } else {
                0
            };
            let used_gb = used as f64 / (1024.0 * 1024.0 * 1024.0);
            let total_gb = peak as f64 / (1024.0 * 1024.0 * 1024.0);
            let color = if pct > 90 {
                "#22c55e"
            } else if pct > 70 {
                "#eab308"
            } else {
                "#ef4444"
            };

            gpu_html.push_str(&format!(
                r#"<div class="gpu-card">
                    <div class="gpu-label">{key} &mdash; {used_gb:.1}/{total_gb:.1} GB</div>
                    <div class="gpu-bar-bg"><div class="gpu-bar" style="width:{pct}%;background:{color}"></div></div>
                    <div class="gpu-pct">{pct}%</div>
                </div>"#
            ));
        }
    }

    format!(
        r#"<div id="gpu-section">{gpu_html}</div>
        <div id="telemetry-section">
            <table>
                <tr><td>CUDA launch p50</td><td>{cuda_p50:.0} us</td></tr>
                <tr><td>CUDA launch p99</td><td>{cuda_p99:.0} us</td></tr>
                <tr><td>NCCL allreduce p50</td><td>{nccl_p50:.0} ms</td></tr>
                <tr><td>NCCL allreduce p99</td><td>{nccl_p99:.0} ms</td></tr>
                <tr><td>DataLoader wait p50</td><td>{dl_p50:.0} ms</td></tr>
            </table>
        </div>"#
    )
}
