// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;

/// CUDA kernel launch event from BPF uprobe on cuLaunchKernel.
#[derive(Debug, Serialize)]
pub struct CudaLaunchEvent {
    pub pid: u32,
    pub kernel_name_hash: u64,
    pub launch_latency_ns: u64,
    pub pcie_transfer_bytes: u64,
    pub transfer_direction: TransferDirection,
}

#[derive(Debug, Serialize)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

pub struct CudaTraceConsumer;

impl CudaTraceConsumer {
    pub fn new() -> Self {
        Self
    }

    pub fn process_event(&self, _raw: &[u8]) -> Option<CudaLaunchEvent> {
        // TODO: Deserialize from BPF ring buffer
        None
    }
}
