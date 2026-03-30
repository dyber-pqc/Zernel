// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;

/// Distributed training synchronization event.
/// Layout must match struct zernel_dist_sync_event in common.h.
#[derive(Debug, Clone, Serialize)]
#[repr(C)]
pub struct DistSyncEvent {
    pub pid: u32,
    pub tid: u32,
    pub wait_ns: u64,
    pub timestamp_ns: u64,
}

pub struct DistSyncConsumer;

impl DistSyncConsumer {
    pub fn new() -> Self {
        Self
    }

    pub fn process_event(&self, raw: &[u8]) -> Option<DistSyncEvent> {
        if raw.len() < std::mem::size_of::<DistSyncEvent>() {
            return None;
        }
        let event = unsafe { &*(raw.as_ptr() as *const DistSyncEvent) };
        Some(event.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_dist_sync_event() {
        let event = DistSyncEvent {
            pid: 42,
            tid: 100,
            wait_ns: 500_000,
            timestamp_ns: 1_000_000_000,
        };

        let raw = unsafe {
            std::slice::from_raw_parts(
                &event as *const DistSyncEvent as *const u8,
                std::mem::size_of::<DistSyncEvent>(),
            )
        };

        let consumer = DistSyncConsumer::new();
        let result = consumer.process_event(raw).unwrap();
        assert_eq!(result.pid, 42);
        assert_eq!(result.wait_ns, 500_000);
    }
}
