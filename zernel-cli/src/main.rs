// Copyright (C) 2026 Dyber, Inc. — Proprietary

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
  zernel gpu status              GPU management
  zernel bench all               Run ML benchmark suite
  zernel debug why-slow          Diagnose training bottlenecks
  zernel data profile ./data     Dataset statistics
  zernel cluster status          Cluster overview
  zernel serve start ./model     Start inference server
  zernel exp list                List all experiments
  zernel model save ./ckpt       Save a model checkpoint
  zernel job submit train.py     Submit distributed training
  zernel doctor                  Diagnose environment
  zernel query \"SELECT ...\"      Query with ZQL")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new ML project
    Init { name: String },
    /// Run a training script with automatic telemetry and experiment tracking
    Run {
        script: String,
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Live dashboard — GPU utilization, training metrics, eBPF telemetry
    Watch,
    /// Diagnose environment issues
    Doctor,
    /// GPU management — top, mem, kill, lock, health
    #[command(subcommand)]
    Gpu(commands::gpu::GpuCommands),
    /// ML benchmark suite — gpu, nccl, dataloader, memory, e2e
    #[command(subcommand)]
    Bench(commands::bench::BenchCommands),
    /// ML training debugger — why-slow, oom, nan, hang
    #[command(subcommand)]
    Debug(commands::debug::DebugCommands),
    /// Experiment tracking
    #[command(subcommand)]
    Exp(commands::exp::ExpCommands),
    /// Show training logs
    Log {
        #[arg(long)]
        id: Option<String>,
        #[arg(long, short)]
        follow: bool,
        #[arg(long)]
        grep: Option<String>,
    },
    /// Dataset management — profile, split, cache, shard
    #[command(subcommand)]
    Data(commands::data::DataCommands),
    /// Model registry
    #[command(subcommand)]
    Model(commands::model::ModelCommands),
    /// Unified inference server — start, stop, benchmark
    #[command(subcommand)]
    Serve(commands::serve::ServeCommands),
    /// Private model & dataset hub
    #[command(subcommand)]
    Hub(commands::hub::HubCommands),
    /// Distributed job management
    #[command(subcommand)]
    Job(commands::job::JobCommands),
    /// GPU cluster management — add, status, sync, run, drain
    #[command(subcommand)]
    Cluster(commands::cluster::ClusterCommands),
    /// Environment management — snapshot, diff, reproduce, export
    #[command(subcommand)]
    Env(commands::env::EnvCommands),
    /// Smart GPU power management & energy tracking
    #[command(subcommand)]
    Power(commands::power::PowerCommands),
    /// Training optimizations — precision, memory, checkpoints, NUMA
    #[command(subcommand)]
    Optimize(commands::optimize::OptimizeCommands),
    /// GPU fleet management — cost attribution, idle detection, capacity planning
    #[command(subcommand)]
    Fleet(commands::fleet::FleetCommands),
    /// Compliance audit trail — lineage, provenance, HIPAA/SOC2 exports
    #[command(subcommand)]
    Audit(commands::audit::AuditCommands),
    /// Developer onboarding — one-command setup, env sync, sharing
    #[command(subcommand)]
    Onboard(commands::onboard::OnboardCommands),
    /// Post-Quantum Cryptography — sign, verify, encrypt, decrypt
    #[command(subcommand)]
    Pqc(commands::pqc::PqcCommands),
    /// GPU cost tracking — summary, budget, report
    #[command(subcommand)]
    Cost(commands::cost::CostCommands),
    /// Jupyter notebook management
    #[command(subcommand)]
    Notebook(commands::notebook::NotebookCommands),
    /// Query experiments, jobs, models with ZQL
    Query { query: String },
    /// Install ML tools (pytorch, ollama, jupyter, etc.)
    Install { tool: String },
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
        Commands::Doctor => commands::doctor::run().await,
        Commands::Gpu(cmd) => commands::gpu::run(cmd).await,
        Commands::Bench(cmd) => commands::bench::run(cmd).await,
        Commands::Debug(cmd) => commands::debug::run(cmd).await,
        Commands::Exp(cmd) => commands::exp::run(cmd).await,
        Commands::Log { id, follow, grep } => commands::log::run(id, follow, grep).await,
        Commands::Data(cmd) => commands::data::run(cmd).await,
        Commands::Model(cmd) => commands::model::run(cmd).await,
        Commands::Serve(cmd) => commands::serve::run(cmd).await,
        Commands::Hub(cmd) => commands::hub::run(cmd).await,
        Commands::Job(cmd) => commands::job::run(cmd).await,
        Commands::Cluster(cmd) => commands::cluster::run(cmd).await,
        Commands::Env(cmd) => commands::env::run(cmd).await,
        Commands::Cost(cmd) => commands::cost::run(cmd).await,
        Commands::Notebook(cmd) => commands::notebook::run(cmd).await,
        Commands::Power(cmd) => commands::power::run(cmd).await,
        Commands::Optimize(cmd) => commands::optimize::run(cmd).await,
        Commands::Pqc(cmd) => commands::pqc::run(cmd).await,
        Commands::Fleet(cmd) => commands::fleet::run(cmd).await,
        Commands::Audit(cmd) => commands::audit::run(cmd).await,
        Commands::Onboard(cmd) => commands::onboard::run(cmd).await,
        Commands::Query { query } => {
            let result = zql::executor::execute(&query)?;
            println!("{result}");
            Ok(())
        }
        Commands::Install { tool } => {
            let status = std::process::Command::new("zernel-install")
                .arg(&tool)
                .status();
            match status {
                Ok(s) if s.success() => Ok(()),
                _ => {
                    println!("zernel-install not in PATH. Run: sudo cp distro/scripts/zernel-install /usr/local/bin/");
                    Ok(())
                }
            }
        }
    }
}
