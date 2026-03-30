// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::Result;
use tracing::info;

/// Launch the full-screen Ratatui dashboard.
pub async fn run() -> Result<()> {
    info!("starting watch dashboard");

    // TODO:
    // 1. Connect to zerneld WebSocket (localhost:9092)
    // 2. Initialize Ratatui terminal with crossterm backend
    // 3. Render dashboard layout:
    //    - GPU utilization bars per device
    //    - GPU memory usage per device
    //    - Training metrics (loss, grad_norm, step, ETA)
    //    - eBPF telemetry (CUDA latency, NCCL duration, DataLoader wait)
    //    - Scheduler phase indicator
    // 4. Update on each WebSocket message
    // 5. Handle keyboard input (q = quit, tab = switch panels)

    println!("zernel watch");
    println!();
    println!("(dashboard not yet implemented — coming in Phase 3)");
    println!("In the meantime, use: watch -n 1 nvidia-smi");

    Ok(())
}
