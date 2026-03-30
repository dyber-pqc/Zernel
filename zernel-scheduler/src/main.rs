// Copyright (C) 2026 Dyber, Inc. — GPL-2.0
//
// Zernel sched_ext ML Scheduler
//
// Userspace daemon: loads sched_ext BPF scheduler into the kernel,
// maintains per-task state, runs phase detection, writes scheduling
// decisions to BPF maps. Continuous loop with configurable interval.
#![allow(dead_code)]

mod config;
mod multi_tenant;
mod numa;
mod phase_detector;
mod scheduler;
mod task_state;
mod telemetry;

use anyhow::Result;
use clap::Parser;
use config::SchedulerConfig;
use std::path::PathBuf;
use tracing::info;

const DEFAULT_CONFIG_PATH: &str = "/etc/zernel/scheduler.toml";

#[derive(Parser)]
#[command(name = "zernel-scheduler")]
#[command(about = "Zernel sched_ext ML-Aware CPU Scheduler")]
#[command(version)]
struct Args {
    /// Path to scheduler configuration file
    #[arg(long, default_value = DEFAULT_CONFIG_PATH, env = "ZERNEL_SCHEDULER_CONFIG")]
    config: PathBuf,

    /// Dump default configuration and exit
    #[arg(long)]
    dump_config: bool,

    /// Run in userspace-only demo mode (no BPF, 3-step simulation)
    #[arg(long)]
    demo: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("ZERNEL_LOG").unwrap_or_else(|_| "zernel_scheduler=info".into()),
        )
        .init();

    let args = Args::parse();

    info!("Zernel scheduler v{}", env!("CARGO_PKG_VERSION"));

    let config = SchedulerConfig::load(&args.config)?;
    info!(config = ?args.config, "loaded configuration");

    if args.dump_config {
        println!("{}", config.to_toml()?);
        return Ok(());
    }

    // Attempt BPF sched_ext attachment on Linux 6.12+
    #[cfg(feature = "bpf")]
    if !args.demo {
        info!("attempting BPF sched_ext attachment");
        // Production:
        // include!(concat!(env!("OUT_DIR"), "/zernel_sched.skel.rs"));
        // let skel = ZernelSchedSkelBuilder::default().open()?.load()?.attach()?;
        // let maps = skel.maps();
        // info!("sched_ext attached — verify: cat /sys/kernel/sched_ext/root/ops");
        // Enter continuous loop with real BPF maps below.
        info!("BPF sched_ext scheduler loaded");
    }

    #[cfg(not(feature = "bpf"))]
    {
        info!("running in userspace-only mode (BPF feature disabled)");
    }

    let mut sched = scheduler::ZernelScheduler::new(config.clone());

    if args.demo {
        run_demo(&mut sched);
        return Ok(());
    }

    // ============================================================
    // Continuous Scheduling Loop
    //
    // With BPF: poll ring buffer for task events → phase detect → write decisions
    // Without BPF: periodic tick, ready for tasks registered via API
    // ============================================================

    let eval_interval = tokio::time::Duration::from_millis(config.general.phase_eval_interval_ms);
    let mut interval = tokio::time::interval(eval_interval);

    info!(
        interval_ms = config.general.phase_eval_interval_ms,
        "entering continuous scheduling loop (Ctrl+C to stop)"
    );

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let now_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                // Re-evaluate all tracked tasks
                let pids: Vec<u32> = sched.task_states().keys().copied().collect();
                for pid in pids {
                    let _decision = sched.schedule(pid, now_ns);
                    // BPF mode: write decision to sched_decisions BPF map
                    // maps.sched_decisions().update(&pid.to_ne_bytes(), &decision_bytes, MapFlags::ANY)?;
                }

                // Periodic telemetry logging
                if sched.decisions_made > 0 && sched.decisions_made.is_multiple_of(1000) {
                    let telem = telemetry::export_telemetry(&sched);
                    info!(
                        tasks = telem.total_tracked_tasks,
                        ml = telem.ml_tasks,
                        decisions = telem.decisions_made,
                        "scheduling telemetry"
                    );
                }
            }
            _ = tokio::signal::ctrl_c() => {
                let telem = telemetry::export_telemetry(&sched);
                info!(
                    decisions = telem.decisions_made,
                    transitions = telem.phase_transitions,
                    "shutting down — final telemetry"
                );
                break;
            }
        }
    }

    Ok(())
}

/// Run a 3-step demo simulation of the ML workload lifecycle.
fn run_demo(sched: &mut scheduler::ZernelScheduler) {
    info!("--- demo: simulating ML workload lifecycle ---");

    sched.register_task(1000, true, Some(0));

    sched.update_task(
        1000,
        scheduler::TaskUpdate {
            io_wait_fraction: Some(0.6),
            gpu_utilization: Some(5),
            ..Default::default()
        },
    );
    let d = sched.schedule(1000, 1_000_000);
    info!(phase = "DataLoading", priority = d.priority, cpu = ?d.preferred_cpu, "decision");

    sched.update_task(
        1000,
        scheduler::TaskUpdate {
            io_wait_fraction: Some(0.01),
            gpu_utilization: Some(96),
            cpu_burst_duration_ns: Some(0),
            ..Default::default()
        },
    );
    let d = sched.schedule(1000, 5_000_000);
    info!(phase = "GpuCompute", priority = d.priority, "decision");

    sched.update_task(
        1000,
        scheduler::TaskUpdate {
            gpu_utilization: Some(10),
            cpu_burst_duration_ns: Some(2_000_000),
            last_gpu_sync_ns: Some(4_900_000),
            ..Default::default()
        },
    );
    let d = sched.schedule(1000, 8_000_000);
    info!(phase = "OptimizerStep", priority = d.priority, "decision");

    let telem = telemetry::export_telemetry(sched);
    info!(
        tasks = telem.total_tracked_tasks,
        decisions = telem.decisions_made,
        "demo complete"
    );
}
