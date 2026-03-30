// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::scheduler::ZernelScheduler;
use crate::task_state::WorkloadPhase;
use serde::Serialize;

/// Snapshot of scheduler state for export to the eBPF observability layer.
#[derive(Debug, Serialize)]
pub struct SchedulerTelemetry {
    pub total_tracked_tasks: usize,
    pub ml_tasks: usize,
    pub decisions_made: u64,
    pub phase_transitions: u64,
    pub phase_distribution: PhaseDistribution,
    pub phase_time_pct: PhaseTimePct,
    pub numa_nodes: usize,
    pub total_cpus: usize,
}

#[derive(Debug, Default, Serialize)]
pub struct PhaseDistribution {
    pub data_loading: usize,
    pub gpu_compute: usize,
    pub nccl_collective: usize,
    pub optimizer_step: usize,
    pub unknown: usize,
}

/// Aggregate phase time percentages across all ML tasks.
#[derive(Debug, Default, Serialize)]
pub struct PhaseTimePct {
    pub data_loading: f64,
    pub gpu_compute: f64,
    pub nccl_collective: f64,
    pub optimizer_step: f64,
    pub unknown: f64,
}

/// Export current scheduler telemetry as a serializable snapshot.
pub fn export_telemetry(scheduler: &ZernelScheduler) -> SchedulerTelemetry {
    let states = scheduler.task_states();
    let mut dist = PhaseDistribution::default();

    let mut ml_count = 0;
    let mut total_data_loading = 0u64;
    let mut total_gpu_compute = 0u64;
    let mut total_nccl = 0u64;
    let mut total_optimizer = 0u64;
    let mut total_unknown = 0u64;

    for state in states.values() {
        if state.is_ml_process {
            ml_count += 1;
            total_data_loading += state.phase_time_ns.data_loading_ns;
            total_gpu_compute += state.phase_time_ns.gpu_compute_ns;
            total_nccl += state.phase_time_ns.nccl_collective_ns;
            total_optimizer += state.phase_time_ns.optimizer_step_ns;
            total_unknown += state.phase_time_ns.unknown_ns;
        }
        match state.current_phase {
            WorkloadPhase::DataLoading => dist.data_loading += 1,
            WorkloadPhase::GpuCompute => dist.gpu_compute += 1,
            WorkloadPhase::NcclCollective => dist.nccl_collective += 1,
            WorkloadPhase::OptimizerStep => dist.optimizer_step += 1,
            WorkloadPhase::Unknown => dist.unknown += 1,
        }
    }

    let total_time = total_data_loading + total_gpu_compute + total_nccl + total_optimizer + total_unknown;
    let pct = if total_time > 0 {
        PhaseTimePct {
            data_loading: total_data_loading as f64 / total_time as f64 * 100.0,
            gpu_compute: total_gpu_compute as f64 / total_time as f64 * 100.0,
            nccl_collective: total_nccl as f64 / total_time as f64 * 100.0,
            optimizer_step: total_optimizer as f64 / total_time as f64 * 100.0,
            unknown: total_unknown as f64 / total_time as f64 * 100.0,
        }
    } else {
        PhaseTimePct::default()
    };

    let numa = scheduler.numa_topology();

    SchedulerTelemetry {
        total_tracked_tasks: states.len(),
        ml_tasks: ml_count,
        decisions_made: scheduler.decisions_made,
        phase_transitions: scheduler.phase_transition_count(),
        phase_distribution: dist,
        phase_time_pct: pct,
        numa_nodes: numa.nodes.len(),
        total_cpus: numa.total_cpus(),
    }
}

/// Format telemetry as Prometheus text exposition.
pub fn format_prometheus(telem: &SchedulerTelemetry) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "# HELP zernel_scheduler_tasks Total tracked tasks\n\
         # TYPE zernel_scheduler_tasks gauge\n\
         zernel_scheduler_tasks{{type=\"total\"}} {}\n\
         zernel_scheduler_tasks{{type=\"ml\"}} {}\n",
        telem.total_tracked_tasks, telem.ml_tasks,
    ));

    out.push_str(&format!(
        "# HELP zernel_scheduler_decisions_total Total scheduling decisions\n\
         # TYPE zernel_scheduler_decisions_total counter\n\
         zernel_scheduler_decisions_total {}\n",
        telem.decisions_made,
    ));

    out.push_str(&format!(
        "# HELP zernel_scheduler_phase_transitions_total Phase transition count\n\
         # TYPE zernel_scheduler_phase_transitions_total counter\n\
         zernel_scheduler_phase_transitions_total {}\n",
        telem.phase_transitions,
    ));

    out.push_str(&format!(
        "# HELP zernel_scheduler_phase_tasks Current tasks per phase\n\
         # TYPE zernel_scheduler_phase_tasks gauge\n\
         zernel_scheduler_phase_tasks{{phase=\"data_loading\"}} {}\n\
         zernel_scheduler_phase_tasks{{phase=\"gpu_compute\"}} {}\n\
         zernel_scheduler_phase_tasks{{phase=\"nccl_collective\"}} {}\n\
         zernel_scheduler_phase_tasks{{phase=\"optimizer_step\"}} {}\n\
         zernel_scheduler_phase_tasks{{phase=\"unknown\"}} {}\n",
        telem.phase_distribution.data_loading,
        telem.phase_distribution.gpu_compute,
        telem.phase_distribution.nccl_collective,
        telem.phase_distribution.optimizer_step,
        telem.phase_distribution.unknown,
    ));

    out.push_str(&format!(
        "# HELP zernel_scheduler_phase_time_pct Aggregate phase time percentage\n\
         # TYPE zernel_scheduler_phase_time_pct gauge\n\
         zernel_scheduler_phase_time_pct{{phase=\"data_loading\"}} {:.2}\n\
         zernel_scheduler_phase_time_pct{{phase=\"gpu_compute\"}} {:.2}\n\
         zernel_scheduler_phase_time_pct{{phase=\"nccl_collective\"}} {:.2}\n\
         zernel_scheduler_phase_time_pct{{phase=\"optimizer_step\"}} {:.2}\n\
         zernel_scheduler_phase_time_pct{{phase=\"unknown\"}} {:.2}\n",
        telem.phase_time_pct.data_loading,
        telem.phase_time_pct.gpu_compute,
        telem.phase_time_pct.nccl_collective,
        telem.phase_time_pct.optimizer_step,
        telem.phase_time_pct.unknown,
    ));

    out
}
