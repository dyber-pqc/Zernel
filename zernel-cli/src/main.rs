// Copyright (C) 2026 Dyber, Inc. — Proprietary

mod commands;
mod experiments;
mod telemetry;
mod zql;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "zernel")]
#[command(about = "Zernel — AI-Native ML Developer Environment")]
#[command(version)]
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
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("zernel=info")
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
    }
}
