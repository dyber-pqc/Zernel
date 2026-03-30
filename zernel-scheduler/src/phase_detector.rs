// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::task_state::{WorkloadPhase, ZernelTaskState};
use serde::{Deserialize, Serialize};

/// Thresholds for phase detection heuristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseDetectorConfig {
    /// I/O wait fraction above which we classify as DataLoading.
    pub io_wait_threshold: f32,
    /// Duration (ns) below which a CPU burst after GpuCompute is an OptimizerStep.
    pub optimizer_burst_max_ns: u64,
    /// GPU utilization below which we suspect the GPU is idle.
    pub gpu_idle_threshold: u8,
    /// GPU utilization above which we consider GPU actively computing.
    pub gpu_active_threshold: u8,
    /// Number of consecutive identical phase samples before committing transition.
    pub phase_stability_count: u32,
    /// Enable NCCL collective detection.
    pub nccl_detection_enabled: bool,
}

impl Default for PhaseDetectorConfig {
    fn default() -> Self {
        Self {
            io_wait_threshold: 0.3,
            optimizer_burst_max_ns: 5_000_000, // 5ms
            gpu_idle_threshold: 10,
            gpu_active_threshold: 80,
            phase_stability_count: 3,
            nccl_detection_enabled: false,
        }
    }
}

/// Tracks per-task phase stability to avoid flapping.
#[derive(Debug, Default)]
struct PhaseStabilityTracker {
    /// The candidate phase we're observing.
    candidate: Option<WorkloadPhase>,
    /// How many consecutive times we've seen this candidate.
    count: u32,
}

/// Detects the current workload phase for a given task based on runtime metrics.
pub struct PhaseDetector {
    config: PhaseDetectorConfig,
    /// Per-pid stability tracking to debounce phase transitions.
    stability: std::collections::HashMap<u32, PhaseStabilityTracker>,
    /// Phase transition counters for telemetry.
    pub transition_count: u64,
}

impl PhaseDetector {
    pub fn new(config: PhaseDetectorConfig) -> Self {
        Self {
            config,
            stability: std::collections::HashMap::new(),
            transition_count: 0,
        }
    }

    /// Classify the workload phase based on current task state.
    /// Uses stability tracking to prevent rapid phase flapping.
    pub fn detect(&mut self, state: &ZernelTaskState) -> WorkloadPhase {
        let raw_phase = self.detect_raw(state);

        if self.config.phase_stability_count <= 1 {
            // Stability disabled — immediate transitions
            if raw_phase != state.current_phase {
                self.transition_count += 1;
            }
            return raw_phase;
        }

        let tracker = self.stability.entry(state.pid).or_default();

        if tracker.candidate == Some(raw_phase) {
            tracker.count += 1;
        } else {
            tracker.candidate = Some(raw_phase);
            tracker.count = 1;
        }

        if tracker.count >= self.config.phase_stability_count {
            if raw_phase != state.current_phase {
                self.transition_count += 1;
            }
            raw_phase
        } else {
            // Not stable yet — keep current phase
            state.current_phase
        }
    }

    /// Raw phase classification without stability tracking.
    fn detect_raw(&self, state: &ZernelTaskState) -> WorkloadPhase {
        if !state.is_ml_process {
            return WorkloadPhase::Unknown;
        }

        // NCCL collective detection (highest priority — on critical path)
        if self.config.nccl_detection_enabled && state.nccl_active && state.futex_wait_count > 0 {
            return WorkloadPhase::NcclCollective;
        }

        // High I/O wait + low GPU util → data loading
        if state.io_wait_fraction > self.config.io_wait_threshold
            && state.gpu_utilization < self.config.gpu_idle_threshold
        {
            return WorkloadPhase::DataLoading;
        }

        // GPU highly utilized + low CPU burst → GPU compute phase
        if state.gpu_utilization > self.config.gpu_active_threshold
            && state.cpu_burst_duration_ns == 0
        {
            return WorkloadPhase::GpuCompute;
        }

        // Short CPU burst right after GPU sync → optimizer step
        if state.cpu_burst_duration_ns > 0
            && state.cpu_burst_duration_ns < self.config.optimizer_burst_max_ns
            && state.last_gpu_sync_ns > 0
        {
            return WorkloadPhase::OptimizerStep;
        }

        // Medium I/O + medium GPU → probably data loading with prefetch overlap
        if state.io_wait_fraction > self.config.io_wait_threshold * 0.5
            && state.gpu_utilization < self.config.gpu_active_threshold
            && state.gpu_utilization > self.config.gpu_idle_threshold
        {
            return WorkloadPhase::DataLoading;
        }

        WorkloadPhase::Unknown
    }

    /// Remove stability tracking for a task.
    pub fn remove_task(&mut self, pid: u32) {
        self.stability.remove(&pid);
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

    fn detector() -> PhaseDetector {
        PhaseDetector::new(PhaseDetectorConfig {
            phase_stability_count: 1, // disable stability for unit tests
            ..Default::default()
        })
    }

    #[test]
    fn non_ml_process_is_unknown() {
        let mut det = detector();
        let state = make_state(1, false);
        assert_eq!(det.detect(&state), WorkloadPhase::Unknown);
    }

    #[test]
    fn high_io_wait_low_gpu_is_data_loading() {
        let mut det = detector();
        let mut state = make_state(1, true);
        state.io_wait_fraction = 0.5;
        state.gpu_utilization = 5;
        assert_eq!(det.detect(&state), WorkloadPhase::DataLoading);
    }

    #[test]
    fn high_gpu_util_is_gpu_compute() {
        let mut det = detector();
        let mut state = make_state(1, true);
        state.gpu_utilization = 95;
        state.cpu_burst_duration_ns = 0;
        assert_eq!(det.detect(&state), WorkloadPhase::GpuCompute);
    }

    #[test]
    fn short_burst_after_sync_is_optimizer_step() {
        let mut det = detector();
        let mut state = make_state(1, true);
        state.cpu_burst_duration_ns = 2_000_000; // 2ms
        state.last_gpu_sync_ns = 1_000_000_000;
        assert_eq!(det.detect(&state), WorkloadPhase::OptimizerStep);
    }

    #[test]
    fn nccl_detection_when_enabled() {
        let mut det = PhaseDetector::new(PhaseDetectorConfig {
            nccl_detection_enabled: true,
            phase_stability_count: 1,
            ..Default::default()
        });
        let mut state = make_state(1, true);
        state.nccl_active = true;
        state.futex_wait_count = 10;
        assert_eq!(det.detect(&state), WorkloadPhase::NcclCollective);
    }

    #[test]
    fn nccl_detection_off_by_default() {
        let mut det = detector();
        let mut state = make_state(1, true);
        state.nccl_active = true;
        state.futex_wait_count = 10;
        // Should NOT detect as NCCL since detection is disabled
        assert_ne!(det.detect(&state), WorkloadPhase::NcclCollective);
    }

    #[test]
    fn stability_prevents_flapping() {
        let mut det = PhaseDetector::new(PhaseDetectorConfig {
            phase_stability_count: 3,
            ..Default::default()
        });

        let mut state = make_state(1, true);
        state.current_phase = WorkloadPhase::Unknown;

        // Set to data loading
        state.io_wait_fraction = 0.5;
        state.gpu_utilization = 5;

        // First two detections should keep Unknown (not stable yet)
        assert_eq!(det.detect(&state), WorkloadPhase::Unknown);
        assert_eq!(det.detect(&state), WorkloadPhase::Unknown);
        // Third detection commits the transition
        assert_eq!(det.detect(&state), WorkloadPhase::DataLoading);
    }

    #[test]
    fn overlapped_prefetch_detected_as_data_loading() {
        let mut det = detector();
        let mut state = make_state(1, true);
        state.io_wait_fraction = 0.2; // moderate I/O
        state.gpu_utilization = 50; // moderate GPU (prefetch overlap)
        assert_eq!(det.detect(&state), WorkloadPhase::DataLoading);
    }

    #[test]
    fn transition_counter_increments() {
        let mut det = detector();
        let mut state = make_state(1, true);
        state.current_phase = WorkloadPhase::Unknown;
        state.io_wait_fraction = 0.5;
        state.gpu_utilization = 5;
        det.detect(&state);
        assert_eq!(det.transition_count, 1);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Phase detection must never panic for any input combination.
        #[test]
        fn detect_never_panics(
            gpu_util in 0u8..=100,
            io_wait in 0.0f32..=1.0,
            cpu_burst in 0u64..=100_000_000,
            gpu_sync in 0u64..=10_000_000_000u64,
            nccl in proptest::bool::ANY,
            futex in 0u32..=1000,
            is_ml in proptest::bool::ANY,
        ) {
            let mut det = PhaseDetector::new(PhaseDetectorConfig {
                phase_stability_count: 1,
                nccl_detection_enabled: true,
                ..Default::default()
            });
            let mut state = ZernelTaskState::new(1);
            state.is_ml_process = is_ml;
            state.gpu_utilization = gpu_util;
            state.io_wait_fraction = io_wait;
            state.cpu_burst_duration_ns = cpu_burst;
            state.last_gpu_sync_ns = gpu_sync;
            state.nccl_active = nccl;
            state.futex_wait_count = futex;

            // Must return a valid phase, never panic
            let phase = det.detect(&state);
            match phase {
                WorkloadPhase::DataLoading
                | WorkloadPhase::GpuCompute
                | WorkloadPhase::NcclCollective
                | WorkloadPhase::OptimizerStep
                | WorkloadPhase::Unknown => {} // all valid
            }
        }

        /// Non-ML processes always get Unknown phase.
        #[test]
        fn non_ml_always_unknown(
            gpu_util in 0u8..=100,
            io_wait in 0.0f32..=1.0,
        ) {
            let mut det = PhaseDetector::new(PhaseDetectorConfig {
                phase_stability_count: 1,
                ..Default::default()
            });
            let mut state = ZernelTaskState::new(1);
            state.is_ml_process = false;
            state.gpu_utilization = gpu_util;
            state.io_wait_fraction = io_wait;

            assert_eq!(det.detect(&state), WorkloadPhase::Unknown);
        }
    }
}
