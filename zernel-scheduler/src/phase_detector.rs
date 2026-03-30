// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::task_state::{WorkloadPhase, ZernelTaskState};

/// Thresholds for phase detection heuristics.
pub struct PhaseDetectorConfig {
    /// I/O wait fraction above which we classify as DataLoading.
    pub io_wait_threshold: f32,
    /// Duration (ns) below which a CPU burst after GpuCompute is an OptimizerStep.
    pub optimizer_burst_max_ns: u64,
    /// GPU utilization below which we suspect the GPU is idle (data starvation).
    pub gpu_idle_threshold: u8,
}

impl Default for PhaseDetectorConfig {
    fn default() -> Self {
        Self {
            io_wait_threshold: 0.3,
            optimizer_burst_max_ns: 5_000_000, // 5ms
            gpu_idle_threshold: 10,
        }
    }
}

/// Detects the current workload phase for a given task based on runtime metrics.
pub struct PhaseDetector {
    config: PhaseDetectorConfig,
}

impl PhaseDetector {
    pub fn new(config: PhaseDetectorConfig) -> Self {
        Self { config }
    }

    /// Classify the workload phase based on current task state.
    ///
    /// Detection heuristics:
    /// - DataLoading: high io_wait fraction + low GPU utilization
    /// - GpuCompute: blocked on cudaDeviceSynchronize (last_gpu_sync recent)
    /// - NcclCollective: detected via NCCL shared memory + high futex activity
    ///   (requires BPF uprobe data — placeholder for now)
    /// - OptimizerStep: short CPU burst immediately after GpuCompute
    pub fn detect(&self, state: &ZernelTaskState) -> WorkloadPhase {
        if !state.is_ml_process {
            return WorkloadPhase::Unknown;
        }

        // High I/O wait + low GPU util → data loading
        if state.io_wait_fraction > self.config.io_wait_threshold
            && state.gpu_utilization < self.config.gpu_idle_threshold
        {
            return WorkloadPhase::DataLoading;
        }

        // GPU highly utilized + low CPU burst → GPU compute phase
        if state.gpu_utilization > 80 && state.cpu_burst_duration_ns == 0 {
            return WorkloadPhase::GpuCompute;
        }

        // Short CPU burst right after GPU sync → optimizer step
        if state.cpu_burst_duration_ns > 0
            && state.cpu_burst_duration_ns < self.config.optimizer_burst_max_ns
            && state.last_gpu_sync_ns > 0
        {
            return WorkloadPhase::OptimizerStep;
        }

        // TODO: NCCL collective detection requires BPF uprobe data on
        // ncclAllReduce/ncclBroadcast and futex activity tracking.
        // Will be implemented when BPF probes are wired up.

        WorkloadPhase::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(pid: u32, ml: bool) -> ZernelTaskState {
        let mut s = ZernelTaskState::new(pid);
        s.is_ml_process = ml;
        s
    }

    #[test]
    fn non_ml_process_is_unknown() {
        let detector = PhaseDetector::new(PhaseDetectorConfig::default());
        let state = make_state(1, false);
        assert_eq!(detector.detect(&state), WorkloadPhase::Unknown);
    }

    #[test]
    fn high_io_wait_low_gpu_is_data_loading() {
        let detector = PhaseDetector::new(PhaseDetectorConfig::default());
        let mut state = make_state(1, true);
        state.io_wait_fraction = 0.5;
        state.gpu_utilization = 5;
        assert_eq!(detector.detect(&state), WorkloadPhase::DataLoading);
    }

    #[test]
    fn high_gpu_util_is_gpu_compute() {
        let detector = PhaseDetector::new(PhaseDetectorConfig::default());
        let mut state = make_state(1, true);
        state.gpu_utilization = 95;
        state.cpu_burst_duration_ns = 0;
        assert_eq!(detector.detect(&state), WorkloadPhase::GpuCompute);
    }

    #[test]
    fn short_burst_after_sync_is_optimizer_step() {
        let detector = PhaseDetector::new(PhaseDetectorConfig::default());
        let mut state = make_state(1, true);
        state.cpu_burst_duration_ns = 2_000_000; // 2ms
        state.last_gpu_sync_ns = 1_000_000_000;
        assert_eq!(detector.detect(&state), WorkloadPhase::OptimizerStep);
    }
}
