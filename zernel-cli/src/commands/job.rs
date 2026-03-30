// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand)]
pub enum JobCommands {
    /// Submit a distributed training job
    Submit {
        /// Script to run
        script: String,
        /// GPUs per node
        #[arg(long, default_value = "1")]
        gpus_per_node: u32,
        /// Number of nodes
        #[arg(long, default_value = "1")]
        nodes: u32,
        /// Framework (pytorch, jax)
        #[arg(long, default_value = "pytorch")]
        framework: String,
        /// Communication backend (nccl, gloo)
        #[arg(long, default_value = "nccl")]
        backend: String,
    },
    /// List running jobs
    List,
    /// Show job status
    Status {
        /// Job ID
        id: String,
    },
    /// Cancel a running job
    Cancel {
        /// Job ID
        id: String,
    },
}

pub async fn run(cmd: JobCommands) -> Result<()> {
    match cmd {
        JobCommands::Submit {
            script,
            gpus_per_node,
            nodes,
            framework,
            backend,
        } => {
            println!("Submitting job: {script}");
            println!("  GPUs/node: {gpus_per_node}");
            println!("  Nodes: {nodes}");
            println!("  Framework: {framework}");
            println!("  Backend: {backend}");
            println!("(not yet implemented)");
        }
        JobCommands::List => {
            println!("No running jobs.");
        }
        JobCommands::Status { id } => {
            println!("Job {id}: (not yet implemented)");
        }
        JobCommands::Cancel { id } => {
            println!("Cancelling job: {id}");
            println!("(not yet implemented)");
        }
    }
    Ok(())
}
