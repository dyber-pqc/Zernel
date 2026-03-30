// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;

/// Distributed training synchronization event.
#[derive(Debug, Serialize)]
pub struct DistSyncEvent {
    pub pid: u32,
    pub rank: u32,
    pub sync_latency_ns: u64,
    pub barrier_wait_ns: u64,
}

pub struct DistSyncConsumer;

impl DistSyncConsumer {
    pub fn new() -> Self {
        Self
    }

    pub fn process_event(&self, _raw: &[u8]) -> Option<DistSyncEvent> {
        None
    }
}
