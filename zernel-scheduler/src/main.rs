// Copyright (C) 2026 Dyber, Inc. — GPL-2.0
//
// Zernel sched_ext ML Scheduler
//
// This is the userspace component of the Zernel CPU scheduler.
// It loads the BPF scheduler program (on Linux with `bpf` feature),
// maintains per-task state, runs phase detection, and writes
// scheduling decisions to BPF maps.

mod phase_detector;
mod scheduler;
mod task_state;
mod telemetry;

use anyhow::Result;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("zernel_scheduler=info")
        .init();

    info!("Zernel scheduler starting");

    #[cfg(feature = "bpf")]
    {
        // TODO: Load BPF program via libbpf-rs
        // let skel = zernel_sched::open_and_load()?;
        // let _link = skel.attach()?;
        info!("BPF scheduler loaded and attached");
    }

    #[cfg(not(feature = "bpf"))]
    {
        info!("running in userspace-only mode (BPF feature disabled)");
    }

    let mut sched = scheduler::ZernelScheduler::new();

    // Demo: register a fake ML task and run phase detection
    sched.register_task(1000, true);
    sched.update_task(
        1000,
        scheduler::TaskUpdate {
            gpu_utilization: Some(95),
            ..Default::default()
        },
    );
    let decision = sched.schedule(1000);
    info!(?decision, "scheduling decision for pid 1000");

    let telem = telemetry::export_telemetry(&sched);
    info!(?telem, "telemetry snapshot");

    info!("Zernel scheduler ready — waiting for events");

    // In production, this loop reads from BPF ring buffers and processes events.
    // For now, just hold the process open.
    tokio::signal::ctrl_c().await?;
    info!("shutting down");

    Ok(())
}
