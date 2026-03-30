// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::Serialize;

/// NCCL collective operation event.
#[derive(Debug, Serialize)]
pub struct NcclEvent {
    pub pid: u32,
    pub op: NcclOp,
    pub size_bytes: u64,
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
}

pub struct NcclConsumer;

impl NcclConsumer {
    pub fn new() -> Self {
        Self
    }

    pub fn process_event(&self, _raw: &[u8]) -> Option<NcclEvent> {
        None
    }
}
