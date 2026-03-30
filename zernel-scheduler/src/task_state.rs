// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::{Deserialize, Serialize};

/// The detected phase of an ML workload.
///
/// The Zernel scheduler uses phase detection to apply different scheduling
/// policies depending on what the ML process is currently doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadPhase {
    /// High I/O, many threads, CPU-intensive preprocessing.
    /// Policy: high priority, aggressive preemption.
    DataLoading,

    /// CPU idle, waiting on GPU compute (cudaDeviceSynchronize).
    /// Policy: very low priority, immediate yield.
    GpuCompute,

    /// Network coordination for collective operations (NCCL).
    /// Policy: high priority, low latency target (<50us).
    NcclCollective,

    /// Short CPU burst after GPU compute — optimizer step.
    /// Policy: high priority, preemptive to minimize GPU idle.
    OptimizerStep,

    /// Unknown phase — fall back to CFS-equivalent behavior.
    Unknown,
}

impl Default for WorkloadPhase {
    fn default() -> Self {
        Self::Unknown
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
    /// GPU utilization percentage (0-100), read from NVIDIA driver interface.
    pub gpu_utilization: u8,
    /// Timestamp (ns) of last cudaDeviceSynchronize call.
    pub last_gpu_sync_ns: u64,
    /// Duration (ns) of the most recent CPU burst.
    pub cpu_burst_duration_ns: u64,
    /// Fraction of time spent in I/O wait (0.0 - 1.0).
    pub io_wait_fraction: f32,
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
        }
    }
}
