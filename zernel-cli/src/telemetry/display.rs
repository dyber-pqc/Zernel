// Copyright (C) 2026 Dyber, Inc. — Proprietary

use super::client::TelemetrySnapshot;

/// Format a utilization bar for terminal display.
pub fn format_gpu_bar(utilization: u8, width: usize) -> String {
    let filled = (utilization as usize * width) / 100;
    let empty = width - filled;
    format!(
        "[{}{}] {}%",
        "#".repeat(filled),
        " ".repeat(empty),
        utilization
    )
}

/// Format the full telemetry snapshot as a multi-line string.
pub fn format_snapshot(snapshot: &TelemetrySnapshot) -> String {
    let mut out = String::new();

    for entry in &snapshot.gpu_utilization {
        let used_gb = entry.current_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_gb = if entry.peak_bytes > 0 {
            entry.peak_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        } else {
            80.0
        };
        out.push_str(&format!(
            "{}: {:.1}/{:.1} GB\n",
            entry.key, used_gb, total_gb,
        ));
    }

    out.push_str(&format!(
        "\nCUDA launch: p50={:.0}us p99={:.0}us\n",
        snapshot.cuda_latency_p50_us, snapshot.cuda_latency_p99_us,
    ));
    out.push_str(&format!(
        "NCCL allreduce: p50={:.0}ms p99={:.0}ms\n",
        snapshot.nccl_allreduce_p50_ms, snapshot.nccl_allreduce_p99_ms,
    ));
    out.push_str(&format!(
        "DataLoader wait: p50={:.0}ms\n",
        snapshot.dataloader_wait_p50_ms,
    ));

    out
}
