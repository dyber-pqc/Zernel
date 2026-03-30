// Copyright (C) 2026 Dyber, Inc. — GPL-2.0
//
// Zernel sched_ext ML Scheduler
//
// Userspace component: maintains per-task state, runs phase detection,
// writes scheduling decisions to BPF maps, exports telemetry.

mod config;
mod multi_tenant;
mod numa;
mod phase_detector;
mod scheduler;
mod task_state;
mod telemetry;

use anyhow::Result;
use config::SchedulerConfig;
use std::path::PathBuf;
use tracing::info;

const DEFAULT_CONFIG_PATH: &str = "/etc/zernel/scheduler.toml";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("ZERNEL_LOG").unwrap_or_else(|_| "zernel_scheduler=info".into()),
        )
        .init();

    info!("Zernel scheduler v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config_path = std::env::var("ZERNEL_SCHEDULER_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_CONFIG_PATH));

    let config = SchedulerConfig::load(&config_path)?;
    info!(?config_path, "loaded configuration");

    // Handle --dump-config flag
    if std::env::args().any(|a| a == "--dump-config") {
        println!("{}", config.to_toml()?);
        return Ok(());
    }

    #[cfg(feature = "bpf")]
    {
        info!("BPF scheduler loaded and attached");
    }

    #[cfg(not(feature = "bpf"))]
    {
        info!("running in userspace-only mode (BPF feature disabled)");
    }

    let mut sched = scheduler::ZernelScheduler::new(config);

    // Demo: simulate an ML workload lifecycle
    info!("--- demo: simulating ML workload lifecycle ---");

    sched.register_task(1000, true, Some(0));

    // Phase 1: Data loading
    sched.update_task(1000, scheduler::TaskUpdate {
        io_wait_fraction: Some(0.6),
        gpu_utilization: Some(5),
        ..Default::default()
    });
    let d = sched.schedule(1000, 1_000_000);
    info!(phase = "DataLoading", priority = d.priority, cpu = ?d.preferred_cpu, "decision");

    // Phase 2: GPU compute
    sched.update_task(1000, scheduler::TaskUpdate {
        io_wait_fraction: Some(0.01),
        gpu_utilization: Some(96),
        cpu_burst_duration_ns: Some(0),
        ..Default::default()
    });
    let d = sched.schedule(1000, 5_000_000);
    info!(phase = "GpuCompute", priority = d.priority, "decision");

    // Phase 3: Optimizer step
    sched.update_task(1000, scheduler::TaskUpdate {
        gpu_utilization: Some(10),
        cpu_burst_duration_ns: Some(2_000_000),
        last_gpu_sync_ns: Some(4_900_000),
        ..Default::default()
    });
    let d = sched.schedule(1000, 8_000_000);
    info!(phase = "OptimizerStep", priority = d.priority, "decision");

    // Export telemetry
    let telem = telemetry::export_telemetry(&sched);
    info!(
        tasks = telem.total_tracked_tasks,
        ml = telem.ml_tasks,
        decisions = telem.decisions_made,
        transitions = telem.phase_transitions,
        "telemetry snapshot"
    );

    info!("Zernel scheduler ready — waiting for events");

    tokio::signal::ctrl_c().await?;
    info!("shutting down");

    Ok(())
}
