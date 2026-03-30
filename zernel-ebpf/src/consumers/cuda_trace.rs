// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;

/// CUDA kernel launch event from BPF uprobe on cuLaunchKernel.
/// Layout must match struct zernel_cuda_event in common.h.
#[derive(Debug, Clone, Serialize)]
#[repr(C)]
pub struct CudaLaunchEvent {
    pub pid: u32,
    _pad: u32,
    pub kernel_hash: u64,
    pub launch_ns: u64,
    pub return_ns: u64,
    pub latency_ns: u64,
}

pub struct CudaTraceConsumer;

impl CudaTraceConsumer {
    pub fn new() -> Self {
        Self
    }

    pub fn process_event(&self, raw: &[u8]) -> Option<CudaLaunchEvent> {
        if raw.len() < std::mem::size_of::<CudaLaunchEvent>() {
            return None;
        }
        let event = unsafe { &*(raw.as_ptr() as *const CudaLaunchEvent) };
        Some(event.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_cuda_event() {
        let event = CudaLaunchEvent {
            pid: 5678,
            _pad: 0,
            kernel_hash: 0xDEADBEEF,
            launch_ns: 1000,
            return_ns: 1142,
            latency_ns: 142,
        };

        let raw = unsafe {
            std::slice::from_raw_parts(
                &event as *const CudaLaunchEvent as *const u8,
                std::mem::size_of::<CudaLaunchEvent>(),
            )
        };

        let consumer = CudaTraceConsumer::new();
        let result = consumer.process_event(raw).unwrap();
        assert_eq!(result.pid, 5678);
        assert_eq!(result.latency_ns, 142);
    }
}
