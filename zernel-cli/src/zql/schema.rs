// Copyright (C) 2026 Dyber, Inc. — Proprietary

// ZQL schema — defines the available tables and their columns.
//
// Tables:
//   experiments — all tracked experiments
//   telemetry  — eBPF telemetry data
//   models     — model registry entries
//   jobs       — distributed training jobs

pub struct TableSchema {
    pub name: &'static str,
    pub columns: &'static [ColumnDef],
}

pub struct ColumnDef {
    pub name: &'static str,
    pub data_type: DataType,
}

pub enum DataType {
    Text,
    Number,
    Timestamp,
}

pub const EXPERIMENTS_SCHEMA: TableSchema = TableSchema {
    name: "experiments",
    columns: &[
        ColumnDef {
            name: "id",
            data_type: DataType::Text,
        },
        ColumnDef {
            name: "name",
            data_type: DataType::Text,
        },
        ColumnDef {
            name: "status",
            data_type: DataType::Text,
        },
        ColumnDef {
            name: "loss",
            data_type: DataType::Number,
        },
        ColumnDef {
            name: "accuracy",
            data_type: DataType::Number,
        },
        ColumnDef {
            name: "learning_rate",
            data_type: DataType::Number,
        },
        ColumnDef {
            name: "batch_size",
            data_type: DataType::Number,
        },
        ColumnDef {
            name: "created_at",
            data_type: DataType::Timestamp,
        },
    ],
};

pub const TELEMETRY_SCHEMA: TableSchema = TableSchema {
    name: "telemetry",
    columns: &[
        ColumnDef {
            name: "job_id",
            data_type: DataType::Text,
        },
        ColumnDef {
            name: "avg_gpu_utilization",
            data_type: DataType::Number,
        },
        ColumnDef {
            name: "dataloader_wait_p99",
            data_type: DataType::Number,
        },
        ColumnDef {
            name: "cuda_launch_p50",
            data_type: DataType::Number,
        },
        ColumnDef {
            name: "nccl_allreduce_p50",
            data_type: DataType::Number,
        },
    ],
};
