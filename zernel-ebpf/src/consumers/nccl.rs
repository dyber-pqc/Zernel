// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;

/// NCCL collective operation event.
/// Layout must match struct zernel_nccl_event in common.h.
#[derive(Debug, Clone, Serialize)]
#[repr(C)]
pub struct NcclEvent {
    pub pid: u32,
    pub op: u8,
    pub _pad: [u8; 3],
    pub size_bytes: u64,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub rank: u32,
    pub num_ranks: u32,
}

#[derive(Debug, Serialize)]
pub enum NcclOp {
    AllReduce,
    Broadcast,
    AllGather,
    ReduceScatter,
    Unknown(u8),
}

impl From<u8> for NcclOp {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::AllReduce,
            1 => Self::Broadcast,
            2 => Self::AllGather,
            3 => Self::ReduceScatter,
            other => Self::Unknown(other),
        }
    }
}

impl NcclEvent {
    pub fn op_name(&self) -> &'static str {
        match self.op {
            0 => "all_reduce",
            1 => "broadcast",
            2 => "all_gather",
            3 => "reduce_scatter",
            _ => "unknown",
        }
    }
}

pub struct NcclConsumer;

impl NcclConsumer {
    pub fn new() -> Self {
        Self
    }

    pub fn process_event(&self, raw: &[u8]) -> Option<NcclEvent> {
        if raw.len() < std::mem::size_of::<NcclEvent>() {
            return None;
        }
        let event = unsafe { &*(raw.as_ptr() as *const NcclEvent) };
        Some(event.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nccl_op_names() {
        let event = NcclEvent {
            pid: 1,
            op: 0,
            _pad: [0; 3],
            size_bytes: 4096,
            start_ns: 0,
            duration_ns: 34_000_000,
            rank: 0,
            num_ranks: 8,
        };
        assert_eq!(event.op_name(), "all_reduce");
    }
}
