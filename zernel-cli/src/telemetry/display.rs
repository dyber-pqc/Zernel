// Copyright (C) 2026 Dyber, Inc. — Proprietary

use super::client::TelemetrySnapshot;

/// Format a telemetry snapshot for terminal display (used by `zernel watch`).
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

    for gpu in &snapshot.gpu_utilization {
        out.push_str(&format!(
            "GPU {} {} Mem: {:.1}/{:.1} GB\n",
            gpu.gpu_id,
            format_gpu_bar(gpu.utilization_pct, 20),
            gpu.memory_used_gb,
            gpu.memory_total_gb,
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
        "DataLoader wait: p50={:.0}ms  PCIe: {:.1} GB/s\n",
        snapshot.dataloader_wait_p50_ms, snapshot.pcie_bandwidth_gbps,
    ));
    out.push_str(&format!("Scheduler phase: {}\n", snapshot.scheduler_phase));

    out
}
