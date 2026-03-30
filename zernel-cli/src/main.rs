// Copyright (C) 2026 Dyber, Inc. — Proprietary

// Allow dead code: telemetry client, ZQL schema, and display helpers are
// public API surface consumed as zerneld integration progresses.
#![allow(dead_code)]

mod commands;
mod experiments;
mod telemetry;
pub mod validation;
mod zql;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "zernel")]
#[command(about = "Zernel — AI-Native ML Developer Environment")]
#[command(version)]
#[command(after_help = "Examples:
  zernel init my-project         Scaffold a new ML project
  zernel run train.py            Run with automatic tracking
  zernel watch                   Live GPU + training dashboard
  zernel exp list                List all experiments
  zernel exp compare exp-a exp-b Diff two experiments
  zernel model save ./ckpt       Save a model checkpoint
  zernel doctor                  Diagnose environment
  zernel query \"SELECT ...\"      Query with ZQL")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new ML project
    Init {
        /// Project name
        name: String,
    },
    /// Run a training script with automatic telemetry and experiment tracking
    Run {
        /// Path to the script to run
        script: String,
        /// Additional arguments to pass to the script
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Live dashboard — GPU utilization, training metrics, eBPF telemetry
    Watch,
    /// Experiment tracking commands
    #[command(subcommand)]
    Exp(commands::exp::ExpCommands),
    /// Model registry commands
    #[command(subcommand)]
    Model(commands::model::ModelCommands),
    /// Distributed job management
    #[command(subcommand)]
    Job(commands::job::JobCommands),
    /// Diagnose environment issues
    Doctor,
    /// Query experiments and telemetry with ZQL
    Query {
        /// ZQL query string
        query: String,
    },
    /// Show training logs for an experiment
    Log {
        /// Experiment ID (default: latest)
        #[arg(long)]
        id: Option<String>,
        /// Follow output in real-time (for running experiments)
        #[arg(long, short)]
        follow: bool,
        /// Filter output lines containing this string
        #[arg(long)]
        grep: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("ZERNEL_LOG").unwrap_or_else(|_| "zernel=warn".into()))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init { name } => commands::init::run(&name).await,
        Commands::Run { script, args } => commands::run::run(&script, &args).await,
        Commands::Watch => commands::watch::run().await,
        Commands::Exp(cmd) => commands::exp::run(cmd).await,
        Commands::Model(cmd) => commands::model::run(cmd).await,
        Commands::Job(cmd) => commands::job::run(cmd).await,
        Commands::Doctor => commands::doctor::run().await,
        Commands::Query { query } => {
            let result = zql::executor::execute(&query)?;
            println!("{result}");
            Ok(())
        }
        Commands::Log { id, follow, grep } => commands::log::run(id, follow, grep).await,
    }
}
