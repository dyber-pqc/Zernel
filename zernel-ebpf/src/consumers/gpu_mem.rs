// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;
use tracing::warn;

/// GPU memory allocation event from BPF uprobe on libcuda.so.
/// Layout must match struct zernel_gpu_mem_event in common.h.
#[derive(Debug, Clone, Serialize)]
#[repr(C)]
pub struct GpuMemEvent {
    pub pid: u32,
    pub gpu_id: u32,
    pub alloc_bytes: u64,
    pub free_bytes: u64,
    pub total_usage: u64,
    pub timestamp_ns: u64,
}

/// Processes GPU memory events from the BPF ring buffer.
pub struct GpuMemConsumer {
    /// OOM warning threshold as fraction of total GPU memory (0.0 - 1.0).
    pub oom_threshold: f64,
    /// Total GPU memory per device in bytes (for OOM calculation).
    pub gpu_total_bytes: u64,
}

impl GpuMemConsumer {
    pub fn new(oom_threshold: f64, gpu_total_bytes: u64) -> Self {
        Self {
            oom_threshold,
            gpu_total_bytes,
        }
    }

    /// Process a raw event from the BPF ring buffer.
    /// Returns the deserialized event, or None if the buffer is too small.
    pub fn process_event(&self, raw: &[u8]) -> Option<GpuMemEvent> {
        let event = deserialize_event::<GpuMemEvent>(raw)?;

        // Check OOM threshold
        if self.gpu_total_bytes > 0 {
            let usage_frac = event.total_usage as f64 / self.gpu_total_bytes as f64;
            if usage_frac > self.oom_threshold {
                warn!(
                    pid = event.pid,
                    gpu_id = event.gpu_id,
                    usage_pct = usage_frac * 100.0,
                    "GPU memory usage exceeds OOM threshold"
                );
            }
        }

        Some(event)
    }
}

/// Safely deserialize a C struct from a raw byte buffer.
/// Returns None if the buffer is too small.
fn deserialize_event<T: Clone>(raw: &[u8]) -> Option<T> {
    if raw.len() < std::mem::size_of::<T>() {
        return None;
    }
    // SAFETY: We verified the buffer is large enough. The BPF ring buffer
    // guarantees the data was written as this exact struct layout (matching
    // #[repr(C)] on the Rust side and the C struct in common.h).
    let event = unsafe { &*(raw.as_ptr() as *const T) };
    Some(event.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_gpu_mem_event() {
        let event = GpuMemEvent {
            pid: 1234,
            gpu_id: 0,
            alloc_bytes: 1024 * 1024,
            free_bytes: 0,
            total_usage: 80 * 1024 * 1024 * 1024,
            timestamp_ns: 999999,
        };

        // Serialize to raw bytes (simulating BPF ring buffer)
        let raw = unsafe {
            std::slice::from_raw_parts(
                &event as *const GpuMemEvent as *const u8,
                std::mem::size_of::<GpuMemEvent>(),
            )
        };

        let consumer = GpuMemConsumer::new(0.95, 80 * 1024 * 1024 * 1024);
        let result = consumer.process_event(raw).unwrap();
        assert_eq!(result.pid, 1234);
        assert_eq!(result.alloc_bytes, 1024 * 1024);
    }

    #[test]
    fn rejects_undersized_buffer() {
        let consumer = GpuMemConsumer::new(0.95, 80 * 1024 * 1024 * 1024);
        assert!(consumer.process_event(&[0u8; 4]).is_none());
    }
}
