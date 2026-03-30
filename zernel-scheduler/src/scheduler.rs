// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::phase_detector::{PhaseDetector, PhaseDetectorConfig};
use crate::task_state::{WorkloadPhase, ZernelTaskState};
use std::collections::HashMap;
use tracing::{debug, info};

/// Scheduling priority assigned to a task based on its detected phase.
#[derive(Debug, Clone, Copy)]
pub struct SchedulingDecision {
    pub priority: i32,
    pub preempt: bool,
    pub latency_target_us: Option<u64>,
}

/// Phase-based scheduling policy table.
///
/// | Phase          | CPU Priority | Preempt    | Latency Target |
/// |----------------|-------------|------------|----------------|
/// | DataLoading    | High (+10)  | Aggressive | Low            |
/// | GpuCompute     | Very Low(-5)| Yield      | —              |
/// | NcclCollective | High (+10)  | Low lat    | <50us          |
/// | OptimizerStep  | High (+10)  | Preemptive | Low            |
/// | Unknown        | Normal (0)  | Standard   | Standard       |
fn policy_for_phase(phase: WorkloadPhase) -> SchedulingDecision {
    match phase {
        WorkloadPhase::DataLoading => SchedulingDecision {
            priority: 10,
            preempt: true,
            latency_target_us: Some(100),
        },
        WorkloadPhase::GpuCompute => SchedulingDecision {
            priority: -5,
            preempt: false,
            latency_target_us: None,
        },
        WorkloadPhase::NcclCollective => SchedulingDecision {
            priority: 10,
            preempt: true,
            latency_target_us: Some(50),
        },
        WorkloadPhase::OptimizerStep => SchedulingDecision {
            priority: 10,
            preempt: true,
            latency_target_us: Some(100),
        },
        WorkloadPhase::Unknown => SchedulingDecision {
            priority: 0,
            preempt: false,
            latency_target_us: None,
        },
    }
}

/// The Zernel ML-aware scheduler.
///
/// In production, this drives the sched_ext BPF scheduler via BPF maps.
/// The userspace component maintains task state, runs phase detection,
/// and writes scheduling decisions back to BPF maps for the kernel to consume.
pub struct ZernelScheduler {
    phase_detector: PhaseDetector,
    task_states: HashMap<u32, ZernelTaskState>,
}

impl ZernelScheduler {
    pub fn new() -> Self {
        Self {
            phase_detector: PhaseDetector::new(PhaseDetectorConfig::default()),
            task_states: HashMap::new(),
        }
    }

    /// Register a new task for tracking.
    pub fn register_task(&mut self, pid: u32, is_ml: bool) {
        let mut state = ZernelTaskState::new(pid);
        state.is_ml_process = is_ml;
        self.task_states.insert(pid, state);
        info!(pid, is_ml, "registered task");
    }

    /// Remove a task from tracking.
    pub fn unregister_task(&mut self, pid: u32) {
        self.task_states.remove(&pid);
        debug!(pid, "unregistered task");
    }

    /// Update task metrics (called from BPF ringbuf events or polling).
    pub fn update_task(&mut self, pid: u32, update: TaskUpdate) {
        if let Some(state) = self.task_states.get_mut(&pid) {
            if let Some(v) = update.gpu_utilization {
                state.gpu_utilization = v;
            }
            if let Some(v) = update.io_wait_fraction {
                state.io_wait_fraction = v;
            }
            if let Some(v) = update.cpu_burst_duration_ns {
                state.cpu_burst_duration_ns = v;
            }
            if let Some(v) = update.last_gpu_sync_ns {
                state.last_gpu_sync_ns = v;
            }
        }
    }

    /// Run phase detection and return a scheduling decision for a task.
    pub fn schedule(&mut self, pid: u32) -> SchedulingDecision {
        let Some(state) = self.task_states.get_mut(&pid) else {
            return policy_for_phase(WorkloadPhase::Unknown);
        };

        let phase = self.phase_detector.detect(state);
        state.current_phase = phase;

        let decision = policy_for_phase(phase);
        debug!(pid, ?phase, priority = decision.priority, "scheduling decision");
        decision
    }

    /// Get current state snapshot for telemetry export.
    pub fn task_states(&self) -> &HashMap<u32, ZernelTaskState> {
        &self.task_states
    }
}

/// Partial update for task metrics.
#[derive(Debug, Default)]
pub struct TaskUpdate {
    pub gpu_utilization: Option<u8>,
    pub io_wait_fraction: Option<f32>,
    pub cpu_burst_duration_ns: Option<u64>,
    pub last_gpu_sync_ns: Option<u64>,
}
