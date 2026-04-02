// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::{Deserialize, Serialize};

/// The detected phase of an ML workload.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkloadPhase {
    /// High I/O, many threads, CPU-intensive preprocessing.
    DataLoading,
    /// CPU idle, waiting on GPU compute (cudaDeviceSynchronize).
    GpuCompute,
    /// Network coordination for collective operations (NCCL).
    NcclCollective,
    /// Short CPU burst after GPU compute — optimizer step.
    OptimizerStep,
    /// Unknown phase — fall back to CFS-equivalent behavior.
    #[default]
    Unknown,
}

// Default derived automatically — Unknown is the first variant via #[default].

impl std::fmt::Display for WorkloadPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataLoading => write!(f, "DataLoading"),
            Self::GpuCompute => write!(f, "GpuCompute"),
            Self::NcclCollective => write!(f, "NcclCollective"),
            Self::OptimizerStep => write!(f, "OptimizerStep"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Per-task state tracked by the Zernel scheduler.
///
/// Maintained in BPF maps at runtime; this Rust struct is the userspace mirror.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZernelTaskState {
    pub pid: u32,
    pub is_ml_process: bool,
    pub current_phase: WorkloadPhase,
    /// GPU utilization percentage (0-100).
    pub gpu_utilization: u8,
    /// Timestamp (ns) of last cudaDeviceSynchronize call.
    pub last_gpu_sync_ns: u64,
    /// Duration (ns) of the most recent CPU burst.
    pub cpu_burst_duration_ns: u64,
    /// Fraction of time spent in I/O wait (0.0 - 1.0).
    pub io_wait_fraction: f32,
    /// Whether NCCL shared memory is mapped for this process.
    pub nccl_active: bool,
    /// Recent futex wait count (high = collective coordination).
    pub futex_wait_count: u32,
    /// GPU ID this task is primarily using (for NUMA affinity).
    pub gpu_id: Option<u32>,
    /// Total time (ns) spent in each phase since tracking started.
    pub phase_time_ns: PhaseTimeAccumulator,
    /// Timestamp (ns) when current phase began.
    pub phase_start_ns: u64,
}

/// Accumulated time spent in each phase for telemetry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhaseTimeAccumulator {
    pub data_loading_ns: u64,
    pub gpu_compute_ns: u64,
    pub nccl_collective_ns: u64,
    pub optimizer_step_ns: u64,
    pub unknown_ns: u64,
}

impl PhaseTimeAccumulator {
    pub fn record(&mut self, phase: WorkloadPhase, duration_ns: u64) {
        match phase {
            WorkloadPhase::DataLoading => self.data_loading_ns += duration_ns,
            WorkloadPhase::GpuCompute => self.gpu_compute_ns += duration_ns,
            WorkloadPhase::NcclCollective => self.nccl_collective_ns += duration_ns,
            WorkloadPhase::OptimizerStep => self.optimizer_step_ns += duration_ns,
            WorkloadPhase::Unknown => self.unknown_ns += duration_ns,
        }
    }

    pub fn total_ns(&self) -> u64 {
        self.data_loading_ns
            + self.gpu_compute_ns
            + self.nccl_collective_ns
            + self.optimizer_step_ns
            + self.unknown_ns
    }

    /// Fraction of time in a given phase (0.0 - 1.0).
    pub fn phase_fraction(&self, phase: WorkloadPhase) -> f64 {
        let total = self.total_ns();
        if total == 0 {
            return 0.0;
        }
        let phase_ns = match phase {
            WorkloadPhase::DataLoading => self.data_loading_ns,
            WorkloadPhase::GpuCompute => self.gpu_compute_ns,
            WorkloadPhase::NcclCollective => self.nccl_collective_ns,
            WorkloadPhase::OptimizerStep => self.optimizer_step_ns,
            WorkloadPhase::Unknown => self.unknown_ns,
        };
        phase_ns as f64 / total as f64
    }
}

impl ZernelTaskState {
    pub fn new(pid: u32) -> Self {
        Self {
            pid,
            is_ml_process: false,
            current_phase: WorkloadPhase::Unknown,
            gpu_utilization: 0,
            last_gpu_sync_ns: 0,
            cpu_burst_duration_ns: 0,
            io_wait_fraction: 0.0,
            nccl_active: false,
            futex_wait_count: 0,
            gpu_id: None,
            phase_time_ns: PhaseTimeAccumulator::default(),
            phase_start_ns: 0,
        }
    }

    /// Transition to a new phase, accumulating time in the old phase.
    pub fn transition_phase(&mut self, new_phase: WorkloadPhase, now_ns: u64) {
        if self.phase_start_ns > 0 && now_ns > self.phase_start_ns {
            let duration = now_ns - self.phase_start_ns;
            self.phase_time_ns.record(self.current_phase, duration);
        }
        self.current_phase = new_phase;
        self.phase_start_ns = now_ns;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_time_accumulation() {
        let mut state = ZernelTaskState::new(1);
        state.transition_phase(WorkloadPhase::DataLoading, 1000);
        state.transition_phase(WorkloadPhase::GpuCompute, 5000);
        state.transition_phase(WorkloadPhase::OptimizerStep, 8000);

        assert_eq!(state.phase_time_ns.data_loading_ns, 4000);
        assert_eq!(state.phase_time_ns.gpu_compute_ns, 3000);
    }

    #[test]
    fn phase_fraction_calculation() {
        let mut acc = PhaseTimeAccumulator::default();
        acc.data_loading_ns = 200;
        acc.gpu_compute_ns = 800;
        assert!((acc.phase_fraction(WorkloadPhase::GpuCompute) - 0.8).abs() < 0.01);
        assert!((acc.phase_fraction(WorkloadPhase::DataLoading) - 0.2).abs() < 0.01);
    }

    #[test]
    fn display_impl() {
        assert_eq!(WorkloadPhase::DataLoading.to_string(), "DataLoading");
        assert_eq!(WorkloadPhase::NcclCollective.to_string(), "NcclCollective");
    }
}
