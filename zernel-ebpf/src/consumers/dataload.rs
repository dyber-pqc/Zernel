// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;

/// Dataset I/O event from DataLoader worker threads.
/// Layout must match struct zernel_dataload_event in common.h.
#[derive(Debug, Clone, Serialize)]
#[repr(C)]
pub struct DataLoadEvent {
    pub pid: u32,
    pub tid: u32,
    pub read_bytes: u64,
    pub latency_ns: u64,
    pub io_type: u8,
    pub _pad: [u8; 7],
}

impl DataLoadEvent {
    pub fn io_type_name(&self) -> &'static str {
        match self.io_type {
            0 => "read",
            1 => "io_uring",
            _ => "unknown",
        }
    }
}

pub struct DataLoadConsumer;

impl DataLoadConsumer {
    pub fn new() -> Self {
        Self
    }

    pub fn process_event(&self, raw: &[u8]) -> Option<DataLoadEvent> {
        if raw.len() < std::mem::size_of::<DataLoadEvent>() {
            return None;
        }
        let event = unsafe { &*(raw.as_ptr() as *const DataLoadEvent) };
        Some(event.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_type_names() {
        let e = DataLoadEvent {
            pid: 1,
            tid: 100,
            read_bytes: 4096,
            latency_ns: 8_000_000,
            io_type: 0,
            _pad: [0; 7],
        };
        assert_eq!(e.io_type_name(), "read");
    }
}
