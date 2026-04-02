// Copyright (C) 2026 Dyber, Inc. — GPL-2.0
//
// Zernel sched_ext ML Scheduler
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
#[cfg(feature = "bpf")]
use libbpf_rs::skel::{SkelBuilder, OpenSkel, Skel};
#[cfg(feature = "bpf")]
use libbpf_rs::MapCore;

const DEFAULT_CONFIG_PATH: &str = "/etc/zernel/scheduler.toml";

#[derive(Parser)]
#[command(name = "zernel-scheduler")]
#[command(about = "Zernel sched_ext ML-Aware CPU Scheduler")]
#[command(version)]
struct Args {
    #[arg(long, default_value = DEFAULT_CONFIG_PATH, env = "ZERNEL_SCHEDULER_CONFIG")]
    config: PathBuf,
    #[arg(long)]
    dump_config: bool,
    #[arg(long)]
    demo: bool,
}

#[cfg(feature = "bpf")]
mod skel {
    include!(concat!(env!("OUT_DIR"), "/zernel_sched.skel.rs"));
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

    // ── BPF sched_ext attachment ──────────────────────────────
    // Keep all BPF objects alive for the entire program lifetime.
    #[cfg(feature = "bpf")]
    let mut _open_object = std::mem::MaybeUninit::uninit();
    #[cfg(feature = "bpf")]
    let _skel_hold;
    #[cfg(feature = "bpf")]
    let _link_hold;

    #[cfg(feature = "bpf")]
    if !args.demo {
        info!("attempting BPF sched_ext attachment");

        let skel_builder = skel::ZernelSchedSkelBuilder::default();
        let open_skel = skel_builder.open(&mut _open_object)
            .expect("failed to open BPF skeleton");
        let mut loaded = open_skel.load()
            .expect("failed to load BPF skeleton");

        // Manually attach struct_ops map (the skeleton attach doesn't handle it)
        let link = loaded.maps.zernel_ops.attach_struct_ops()
            .expect("failed to attach struct_ops scheduler — is CONFIG_SCHED_CLASS_EXT=y?");
        info!("sched_ext scheduler ATTACHED — zernel is now the kernel scheduler");

        // Verify
        if let Ok(state) = std::fs::read_to_string("/sys/kernel/sched_ext/state") {
            info!(state = state.trim(), "sched_ext kernel state");
        }

        _link_hold = Some(link);
        _skel_hold = Some(loaded);
    } else {
        _link_hold = None;
        _skel_hold = None;
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

                let pids: Vec<u32> = sched.task_states().keys().copied().collect();
                for pid in pids {
                    let _decision = sched.schedule(pid, now_ns);
                }

                if sched.decisions_made > 0 && sched.decisions_made % 1000 == 0 {
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

fn run_demo(sched: &mut scheduler::ZernelScheduler) {
    info!("--- demo: simulating ML workload lifecycle ---");

    sched.register_task(1000, true, Some(0));

    sched.update_task(1000, scheduler::TaskUpdate {
        io_wait_fraction: Some(0.6), gpu_utilization: Some(5), ..Default::default()
    });
    let d = sched.schedule(1000, 1_000_000);
    info!(phase = "DataLoading", priority = d.priority, cpu = ?d.preferred_cpu, "decision");

    sched.update_task(1000, scheduler::TaskUpdate {
        io_wait_fraction: Some(0.01), gpu_utilization: Some(96),
        cpu_burst_duration_ns: Some(0), ..Default::default()
    });
    let d = sched.schedule(1000, 5_000_000);
    info!(phase = "GpuCompute", priority = d.priority, "decision");

    sched.update_task(1000, scheduler::TaskUpdate {
        gpu_utilization: Some(10), cpu_burst_duration_ns: Some(2_000_000),
        last_gpu_sync_ns: Some(4_900_000), ..Default::default()
    });
    let d = sched.schedule(1000, 8_000_000);
    info!(phase = "OptimizerStep", priority = d.priority, "decision");

    let telem = telemetry::export_telemetry(sched);
    info!(tasks = telem.total_tracked_tasks, decisions = telem.decisions_made, "demo complete");
}
