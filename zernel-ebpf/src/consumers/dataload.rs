// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;

/// Dataset I/O event from DataLoader worker threads.
#[derive(Debug, Serialize)]
pub struct DataLoadEvent {
    pub pid: u32,
    pub worker_id: u32,
    pub read_bytes: u64,
    pub read_latency_ns: u64,
    pub cache_hit: bool,
}

pub struct DataLoadConsumer;

impl DataLoadConsumer {
    pub fn new() -> Self {
        Self
    }

    pub fn process_event(&self, _raw: &[u8]) -> Option<DataLoadEvent> {
        None
    }
}
