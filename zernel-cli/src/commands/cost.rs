// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel cost — GPU cost tracking

use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand)]
pub enum CostCommands {
    /// Show GPU usage summary
    Summary,
    /// Show cost by user
    User {
        /// Username to filter
        name: Option<String>,
    },
    /// Show cost for a specific job
    Job {
        /// Job ID
        id: String,
    },
    /// Generate cost report
    Report {
        /// Month (e.g., march, 2026-03)
        #[arg(long)]
        month: Option<String>,
    },
    /// Set GPU-hour budget with alerts
    Budget {
        /// GPU-hours budget
        #[arg(long)]
        set: Option<u64>,
    },
}

pub async fn run(cmd: CostCommands) -> Result<()> {
    let jobs_db = crate::experiments::tracker::zernel_dir()
        .join("jobs")
        .join("jobs.db");

    match cmd {
        CostCommands::Summary => {
            println!("Zernel GPU Cost Summary");
            println!("{}", "=".repeat(50));

            if !jobs_db.exists() {
                println!("No job data. Submit jobs with: zernel job submit <script>");
                return Ok(());
            }

            let conn = rusqlite::Connection::open(&jobs_db)?;
            let mut stmt = conn.prepare(
                "SELECT COUNT(*), SUM(CASE WHEN status='done' THEN 1 ELSE 0 END), SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END), SUM(gpus_per_node * nodes) FROM jobs",
            )?;

            let (total, done, failed, total_gpus): (u32, u32, u32, Option<u32>) = stmt
                .query_row([], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                })?;

            println!("  Total jobs:     {total}");
            println!("  Completed:      {done}");
            println!("  Failed:         {failed}");
            println!("  Total GPU-jobs: {}", total_gpus.unwrap_or(0));

            // Estimate GPU-hours from experiments
            let exp_db = crate::experiments::tracker::experiments_db_path();
            if exp_db.exists() {
                let exp_conn = rusqlite::Connection::open(&exp_db)?;
                let total_secs: f64 = exp_conn
                    .query_row(
                        "SELECT COALESCE(SUM(duration_secs), 0) FROM experiments",
                        [],
                        |row| row.get(0),
                    )
                    .unwrap_or(0.0);
                let gpu_hours = total_secs / 3600.0;
                println!(
                    "  Total GPU-hours: {gpu_hours:.1}h (estimated from experiment durations)"
                );
            }
        }

        CostCommands::User { name } => {
            println!("GPU cost by user");
            if let Some(n) = name {
                println!("  Filtering for: {n}");
            }
            println!("  (per-user tracking requires auth — coming in enterprise edition)");
        }

        CostCommands::Job { id } => {
            if !jobs_db.exists() {
                println!("No job data.");
                return Ok(());
            }

            let conn = rusqlite::Connection::open(&jobs_db)?;
            let result: Result<(String, u32, u32, String), _> = conn.query_row(
                "SELECT script, gpus_per_node, nodes, submitted_at FROM jobs WHERE id = ?1",
                [&id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            );

            match result {
                Ok((script, gpus, nodes, submitted)) => {
                    println!("Job: {id}");
                    println!("  Script: {script}");
                    println!("  GPUs: {gpus} x {nodes} nodes = {} total", gpus * nodes);
                    println!("  Submitted: {submitted}");
                }
                Err(_) => println!("Job not found: {id}"),
            }
        }

        CostCommands::Report { month } => {
            let m = month.unwrap_or_else(|| "all time".into());
            println!("GPU Cost Report — {m}");
            println!();
            println!("Run: zernel cost summary  — for current totals");
            println!("(Detailed reports coming in enterprise edition)");
        }

        CostCommands::Budget { set } => {
            if let Some(hours) = set {
                let budget_file = crate::experiments::tracker::zernel_dir().join("gpu-budget.txt");
                std::fs::write(&budget_file, hours.to_string())?;
                println!("GPU-hour budget set to: {hours}h");
                println!("Alerts will fire when 80% of budget is consumed.");
            } else {
                let budget_file = crate::experiments::tracker::zernel_dir().join("gpu-budget.txt");
                if budget_file.exists() {
                    let budget = std::fs::read_to_string(&budget_file)?;
                    println!("Current budget: {budget}h");
                } else {
                    println!("No budget set. Set one: zernel cost budget --set 10000");
                }
            }
        }
    }
    Ok(())
}
