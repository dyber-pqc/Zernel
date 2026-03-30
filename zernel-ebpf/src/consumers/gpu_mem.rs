// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;

/// GPU memory allocation event from BPF uprobe on libcuda.so.
#[derive(Debug, Serialize)]
pub struct GpuMemEvent {
    pub pid: u32,
    pub gpu_id: u32,
    pub allocated_bytes: u64,
    pub freed_bytes: u64,
    pub current_usage_bytes: u64,
    pub peak_usage_bytes: u64,
}

/// Processes GPU memory events from the BPF ring buffer.
pub struct GpuMemConsumer {
    /// OOM warning threshold as fraction of total GPU memory (0.0 - 1.0).
    pub oom_threshold: f64,
}

impl GpuMemConsumer {
    pub fn new(oom_threshold: f64) -> Self {
        Self { oom_threshold }
    }

    /// Process a raw event from the BPF ring buffer.
    pub fn process_event(&self, _raw: &[u8]) -> Option<GpuMemEvent> {
        // TODO: Deserialize from BPF ring buffer format
        // Check against OOM threshold and emit warning if needed
        None
    }
}
