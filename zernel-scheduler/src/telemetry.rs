// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::scheduler::ZernelScheduler;
use crate::task_state::WorkloadPhase;
use serde::Serialize;

/// Snapshot of scheduler state for export to the eBPF observability layer.
#[derive(Debug, Serialize)]
pub struct SchedulerTelemetry {
    pub total_tracked_tasks: usize,
    pub ml_tasks: usize,
    pub phase_distribution: PhaseDistribution,
}

#[derive(Debug, Default, Serialize)]
pub struct PhaseDistribution {
    pub data_loading: usize,
    pub gpu_compute: usize,
    pub nccl_collective: usize,
    pub optimizer_step: usize,
    pub unknown: usize,
}

/// Export current scheduler telemetry as a serializable snapshot.
pub fn export_telemetry(scheduler: &ZernelScheduler) -> SchedulerTelemetry {
    let states = scheduler.task_states();
    let mut dist = PhaseDistribution::default();

    let mut ml_count = 0;
    for state in states.values() {
        if state.is_ml_process {
            ml_count += 1;
        }
        match state.current_phase {
            WorkloadPhase::DataLoading => dist.data_loading += 1,
            WorkloadPhase::GpuCompute => dist.gpu_compute += 1,
            WorkloadPhase::NcclCollective => dist.nccl_collective += 1,
            WorkloadPhase::OptimizerStep => dist.optimizer_step += 1,
            WorkloadPhase::Unknown => dist.unknown += 1,
        }
    }

    SchedulerTelemetry {
        total_tracked_tasks: states.len(),
        ml_tasks: ml_count,
        phase_distribution: dist,
    }
}
