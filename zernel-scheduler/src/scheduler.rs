// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::config::SchedulerConfig;
use crate::multi_tenant::TenantScheduler;
use crate::numa::NumaTopology;
use crate::phase_detector::PhaseDetector;
use crate::task_state::{WorkloadPhase, ZernelTaskState};
use std::collections::HashMap;
use tracing::{debug, info};

/// Scheduling priority assigned to a task based on its detected phase.
#[derive(Debug, Clone, Copy)]
pub struct SchedulingDecision {
    pub priority: i32,
    pub preempt: bool,
    pub latency_target_us: Option<u64>,
    pub preferred_cpu: Option<u32>,
}

/// Phase-based scheduling policy table.
fn policy_for_phase(phase: WorkloadPhase) -> (i32, bool, Option<u64>) {
    match phase {
        WorkloadPhase::DataLoading => (10, true, Some(100)),
        WorkloadPhase::GpuCompute => (-5, false, None),
        WorkloadPhase::NcclCollective => (10, true, Some(50)),
        WorkloadPhase::OptimizerStep => (10, true, Some(100)),
        WorkloadPhase::Unknown => (0, false, None),
    }
}

/// The Zernel ML-aware scheduler.
///
/// Integrates phase detection, NUMA-aware CPU selection, and
/// multi-tenant GPU-proportional scheduling.
pub struct ZernelScheduler {
    config: SchedulerConfig,
    phase_detector: PhaseDetector,
    task_states: HashMap<u32, ZernelTaskState>,
    numa: NumaTopology,
    tenant_scheduler: TenantScheduler,
    /// Per-CPU load estimates (cpu_id -> load fraction 0.0-1.0).
    cpu_loads: HashMap<u32, f32>,
    /// Total scheduling decisions made.
    pub decisions_made: u64,
}

impl ZernelScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let phase_config = (&config.phase_detection).into();
        let numa = if config.numa.enabled {
            NumaTopology::detect()
        } else {
            NumaTopology::detect() // still detect, just don't use for affinity
        };

        info!(
            numa_nodes = numa.nodes.len(),
            total_cpus = numa.total_cpus(),
            gpus_mapped = numa.gpu_node_map.len(),
            "NUMA topology detected"
        );

        Self {
            config,
            phase_detector: PhaseDetector::new(phase_config),
            task_states: HashMap::new(),
            numa,
            tenant_scheduler: TenantScheduler::new(),
            cpu_loads: HashMap::new(),
            decisions_made: 0,
        }
    }

    /// Register a new task for tracking.
    pub fn register_task(&mut self, pid: u32, is_ml: bool, gpu_id: Option<u32>) {
        let mut state = ZernelTaskState::new(pid);
        state.is_ml_process = is_ml;
        state.gpu_id = gpu_id;
        self.task_states.insert(pid, state);
        info!(pid, is_ml, ?gpu_id, "registered task");
    }

    /// Remove a task from tracking.
    pub fn unregister_task(&mut self, pid: u32) {
        self.task_states.remove(&pid);
        self.phase_detector.remove_task(pid);
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
            if let Some(v) = update.nccl_active {
                state.nccl_active = v;
            }
            if let Some(v) = update.futex_wait_count {
                state.futex_wait_count = v;
            }
            if let Some(v) = update.gpu_id {
                state.gpu_id = Some(v);
            }
        }
    }

    /// Update CPU load estimates (called periodically from /proc/stat or BPF).
    pub fn update_cpu_loads(&mut self, loads: HashMap<u32, f32>) {
        self.cpu_loads = loads;
    }

    /// Run phase detection and return a scheduling decision for a task.
    pub fn schedule(&mut self, pid: u32, now_ns: u64) -> SchedulingDecision {
        let Some(state) = self.task_states.get_mut(&pid) else {
            return SchedulingDecision {
                priority: 0,
                preempt: false,
                latency_target_us: None,
                preferred_cpu: None,
            };
        };

        // Phase detection
        let new_phase = self.phase_detector.detect(state);
        if new_phase != state.current_phase {
            state.transition_phase(new_phase, now_ns);
        }

        // Base policy from phase
        let (base_priority, preempt, latency_target_us) = policy_for_phase(new_phase);

        // Multi-tenant priority adjustment
        let priority = if self.config.multi_tenant.enabled {
            self.tenant_scheduler.effective_priority(pid, base_priority)
        } else {
            base_priority
        };

        // NUMA-aware CPU selection
        let preferred_cpu = if self.config.numa.gpu_affinity {
            let gpu_id = state.gpu_id;
            Some(self.numa.select_cpu(gpu_id, &self.cpu_loads))
        } else {
            None
        };

        self.decisions_made += 1;

        let decision = SchedulingDecision {
            priority,
            preempt,
            latency_target_us,
            preferred_cpu,
        };

        debug!(
            pid,
            phase = %new_phase,
            priority = decision.priority,
            ?preferred_cpu,
            "scheduling decision"
        );

        decision
    }

    /// Get a reference to the tenant scheduler.
    pub fn tenant_scheduler_mut(&mut self) -> &mut TenantScheduler {
        &mut self.tenant_scheduler
    }

    /// Get current state snapshot for telemetry export.
    pub fn task_states(&self) -> &HashMap<u32, ZernelTaskState> {
        &self.task_states
    }

    pub fn numa_topology(&self) -> &NumaTopology {
        &self.numa
    }

    pub fn phase_transition_count(&self) -> u64 {
        self.phase_detector.transition_count
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

/// Partial update for task metrics.
#[derive(Debug, Default)]
pub struct TaskUpdate {
    pub gpu_utilization: Option<u8>,
    pub io_wait_fraction: Option<f32>,
    pub cpu_burst_duration_ns: Option<u64>,
    pub last_gpu_sync_ns: Option<u64>,
    pub nccl_active: Option<bool>,
    pub futex_wait_count: Option<u32>,
    pub gpu_id: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_sched() -> ZernelScheduler {
        let mut config = SchedulerConfig::default();
        // Disable phase stability in tests for immediate transitions
        config.phase_detection.phase_stability_count = 1;
        ZernelScheduler::new(config)
    }

    #[test]
    fn register_and_schedule() {
        let mut sched = default_sched();
        sched.register_task(100, true, Some(0));
        sched.update_task(100, TaskUpdate {
            gpu_utilization: Some(95),
            ..Default::default()
        });

        let decision = sched.schedule(100, 1000);
        assert_eq!(decision.priority, -5); // GpuCompute
        assert!(!decision.preempt);
    }

    #[test]
    fn data_loading_gets_high_priority() {
        let mut sched = default_sched();
        sched.register_task(100, true, Some(0));
        sched.update_task(100, TaskUpdate {
            io_wait_fraction: Some(0.5),
            gpu_utilization: Some(5),
            ..Default::default()
        });

        let decision = sched.schedule(100, 1000);
        assert_eq!(decision.priority, 10);
        assert!(decision.preempt);
    }

    #[test]
    fn preferred_cpu_set_with_numa() {
        let mut sched = default_sched();
        sched.register_task(100, true, Some(0));
        let decision = sched.schedule(100, 1000);
        assert!(decision.preferred_cpu.is_some());
    }

    #[test]
    fn unknown_pid_gets_default() {
        let mut sched = default_sched();
        let decision = sched.schedule(999, 1000);
        assert_eq!(decision.priority, 0);
    }

    #[test]
    fn decisions_counter_increments() {
        let mut sched = default_sched();
        sched.register_task(100, true, None);
        sched.schedule(100, 1000);
        sched.schedule(100, 2000);
        assert_eq!(sched.decisions_made, 2);
    }
}
