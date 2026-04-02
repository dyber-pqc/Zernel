// Copyright (C) 2026 Dyber, Inc. — GPL-2.0
//
// Zernel sched_ext ML Scheduler (v3)
//
// Full pipeline:
//   1. Load sched_ext BPF scheduler into kernel
//   2. Discover GPU processes (nvidia-smi) and register them
//   3. Poll GPU utilization and update task state
//   4. Run phase detection on tracked tasks
//   5. Write detected phases to BPF phase_map
//   6. Write CPU affinity hints to BPF cpu_affinity_map
//   7. Adjust GPU power profiles per detected phase
//   8. Read ring buffer events for observability
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
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{debug, info, warn};
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

// ── GPU process discovery via nvidia-smi ──────────────────────

/// Discover PIDs currently using NVIDIA GPUs.
fn discover_gpu_processes() -> Vec<(u32, u32)> {
    // (pid, gpu_id) pairs
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"])
        .output();

    let mut results = Vec::new();
    if let Ok(out) = output {
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            for line in stdout.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 2 {
                    if let Ok(pid) = parts[0].parse::<u32>() {
                        // Simple GPU ID: count lines from nvidia-smi pci.bus_id
                        results.push((pid, 0)); // GPU 0 for single-GPU systems
                    }
                }
            }
        }
    }
    results
}

/// Get GPU utilization from nvidia-smi.
fn get_gpu_utilization() -> Option<u8> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().lines().next()?.trim().parse().ok()
}

/// Get GPU power draw in watts.
fn get_gpu_power() -> Option<f32> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=power.draw", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().lines().next()?.trim().parse().ok()
}

/// Get GPU max clocks for power management.
fn get_gpu_max_clocks() -> Option<(u32, u32, u32)> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=clocks.max.graphics,clocks.max.memory,power.max_limit",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let fields: Vec<&str> = stdout.trim().split(',').map(|s| s.trim()).collect();
    if fields.len() >= 3 {
        let g = fields[0].parse().ok()?;
        let m = fields[1].parse().ok()?;
        let p = fields[2].parse::<f32>().ok()? as u32;
        Some((g, m, p))
    } else {
        None
    }
}

/// Apply GPU power profile for a phase.
fn apply_gpu_power_profile(phase: &task_state::WorkloadPhase, max_clocks: (u32, u32, u32)) {
    let (max_g, max_m, max_p) = max_clocks;
    let (target_g, target_p) = match phase {
        task_state::WorkloadPhase::DataLoading => (max_g / 3, (max_p as f32 * 0.6) as u32),
        task_state::WorkloadPhase::GpuCompute => (max_g, max_p),
        task_state::WorkloadPhase::NcclCollective => (max_g / 2, (max_p as f32 * 0.7) as u32),
        task_state::WorkloadPhase::OptimizerStep => (max_g, max_p),
        task_state::WorkloadPhase::Unknown => return,
    };

    // Set power limit
    let _ = std::process::Command::new("nvidia-smi")
        .args(["-i", "0", "-pl", &target_p.to_string()])
        .output();

    // Set application clocks
    let _ = std::process::Command::new("nvidia-smi")
        .args(["-i", "0", "-ac", &format!("{},{}", max_m, target_g)])
        .output();

    debug!(phase = %phase, gpu_clock = target_g, power_limit = target_p, "GPU power profile applied");
}

/// Check if a process is alive.
fn process_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{}", pid)).exists()
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

        let link = loaded.maps.zernel_ops.attach_struct_ops()
            .expect("failed to attach struct_ops scheduler — is CONFIG_SCHED_CLASS_EXT=y?");
        info!("sched_ext scheduler ATTACHED — zernel is now the kernel scheduler");

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

    // ── Discover GPU max clocks for power management ──────────
    let gpu_max_clocks = get_gpu_max_clocks();
    if let Some((g, m, p)) = gpu_max_clocks {
        info!(
            max_graphics = g, max_memory = m, max_power = p,
            "GPU max clocks detected — power management enabled"
        );
        // Enable persistence mode for clock control
        let _ = std::process::Command::new("nvidia-smi")
            .args(["-i", "0", "-pm", "1"])
            .output();
    } else {
        warn!("could not detect GPU max clocks — power management disabled");
    }

    // Track the dominant phase across all GPU tasks for power management
    let mut current_power_phase = task_state::WorkloadPhase::Unknown;
    let mut gpu_poll_counter = 0u64;
    let gpu_poll_interval = config.general.gpu_poll_interval_ms / config.general.phase_eval_interval_ms;

    let eval_interval = tokio::time::Duration::from_millis(config.general.phase_eval_interval_ms);
    let mut interval = tokio::time::interval(eval_interval);

    info!(
        interval_ms = config.general.phase_eval_interval_ms,
        gpu_poll_every = gpu_poll_interval,
        "entering continuous scheduling loop (Ctrl+C to stop)"
    );

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let now_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                gpu_poll_counter += 1;

                // ── Periodic GPU process discovery + metrics update ──
                if gpu_poll_counter % gpu_poll_interval == 0 {
                    // Discover new GPU processes
                    let gpu_procs = discover_gpu_processes();
                    for (pid, gpu_id) in &gpu_procs {
                        if !sched.task_states().contains_key(pid) {
                            sched.register_task(*pid, true, Some(*gpu_id));
                            info!(pid, gpu_id, "discovered GPU process");
                        }
                    }

                    // Clean up dead processes
                    let dead_pids: Vec<u32> = sched.task_states().keys()
                        .filter(|pid| !process_alive(**pid))
                        .copied()
                        .collect();
                    for pid in dead_pids {
                        sched.unregister_task(pid);
                        debug!(pid, "cleaned up dead process");
                    }

                    // Update GPU utilization for all tracked tasks
                    if let Some(gpu_util) = get_gpu_utilization() {
                        let pids: Vec<u32> = sched.task_states().keys().copied().collect();
                        for pid in &pids {
                            sched.update_task(*pid, scheduler::TaskUpdate {
                                gpu_utilization: Some(gpu_util),
                                ..Default::default()
                            });
                        }
                    }
                }

                // ── Run phase detection on all tracked tasks ──
                let pids: Vec<u32> = sched.task_states().keys().copied().collect();
                let mut phase_counts: HashMap<task_state::WorkloadPhase, u32> = HashMap::new();

                for pid in &pids {
                    let decision = sched.schedule(*pid, now_ns);

                    // Get the detected phase
                    if let Some(state) = sched.task_states().get(pid) {
                        *phase_counts.entry(state.current_phase).or_insert(0) += 1;

                        // ── Write phase to BPF phase_map ──
                        #[cfg(feature = "bpf")]
                        if !args.demo {
                            let phase_val: u32 = match state.current_phase {
                                task_state::WorkloadPhase::DataLoading => 0,
                                task_state::WorkloadPhase::GpuCompute => 1,
                                task_state::WorkloadPhase::NcclCollective => 2,
                                task_state::WorkloadPhase::OptimizerStep => 3,
                                task_state::WorkloadPhase::Unknown => 255,
                            };
                            let key = pid.to_ne_bytes();
                            let val = phase_val.to_ne_bytes();
                            let _ = _skel_hold.as_ref().unwrap().maps.phase_map
                                .update(&key, &val, libbpf_rs::MapFlags::ANY);
                        }

                        // ── Write CPU affinity for data-loading tasks ──
                        #[cfg(feature = "bpf")]
                        if let Some(cpu) = decision.preferred_cpu {
                            if state.current_phase == task_state::WorkloadPhase::DataLoading {
                                let key = pid.to_ne_bytes();
                                let val = (cpu as i32).to_ne_bytes();
                                let _ = _skel_hold.as_ref().unwrap().maps.cpu_affinity_map
                                    .update(&key, &val, libbpf_rs::MapFlags::ANY);
                            }
                        }
                    }
                }

                // ─��� GPU power management: apply dominant phase ──
                if let Some(max_clocks) = gpu_max_clocks {
                    // Find the dominant phase among GPU tasks
                    let dominant = phase_counts.iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(phase, _)| *phase)
                        .unwrap_or(task_state::WorkloadPhase::Unknown);

                    if dominant != current_power_phase && dominant != task_state::WorkloadPhase::Unknown {
                        apply_gpu_power_profile(&dominant, max_clocks);
                        info!(
                            from = %current_power_phase, to = %dominant,
                            "GPU power phase transition"
                        );
                        current_power_phase = dominant;
                    }
                }

                // ─�� Periodic telemetry ──
                if sched.decisions_made > 0 && sched.decisions_made % 500 == 0 {
                    let telem = telemetry::export_telemetry(&sched);
                    let power = get_gpu_power().unwrap_or(0.0);
                    info!(
                        tasks = telem.total_tracked_tasks,
                        ml = telem.ml_tasks,
                        decisions = telem.decisions_made,
                        transitions = telem.phase_transitions,
                        gpu_power_w = format!("{:.1}", power),
                        power_phase = %current_power_phase,
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

                // Reset GPU power on exit
                if gpu_max_clocks.is_some() {
                    let _ = std::process::Command::new("nvidia-smi")
                        .args(["-i", "0", "-rac"]).output();
                    let _ = std::process::Command::new("nvidia-smi")
                        .args(["-i", "0", "-rpl"]).output();
                    info!("GPU power reset to defaults");
                }
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
