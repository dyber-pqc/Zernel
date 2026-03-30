// Copyright (C) 2026 Dyber, Inc. — Proprietary

use crate::experiments::compare;
use crate::experiments::store::ExperimentStore;
use crate::experiments::tracker;
use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand)]
pub enum ExpCommands {
    /// List all experiments
    List {
        /// Maximum number of experiments to show
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
    /// Compare two experiments
    Compare {
        /// First experiment ID
        a: String,
        /// Second experiment ID
        b: String,
    },
    /// Show details of an experiment
    Show {
        /// Experiment ID
        id: String,
    },
    /// Delete an experiment
    Delete {
        /// Experiment ID
        id: String,
    },
}

pub async fn run(cmd: ExpCommands) -> Result<()> {
    let db_path = tracker::experiments_db_path();
    let store = ExperimentStore::open(&db_path)?;

    match cmd {
        ExpCommands::List { limit } => {
            let experiments = store.list(limit)?;
            if experiments.is_empty() {
                println!("No experiments yet. Run `zernel run <script>` to create one.");
                return Ok(());
            }

            let header = format!(
                "{:<28} {:<24} {:<10} {:>10} {:>10} {:>10}",
                "ID", "Name", "Status", "Loss", "Acc", "Duration"
            );
            println!("{header}");
            println!("{}", "-".repeat(95));

            for exp in &experiments {
                let loss = exp
                    .metrics
                    .get("loss")
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "-".into());
                let acc = exp
                    .metrics
                    .get("accuracy")
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "-".into());
                let duration = exp
                    .duration_secs
                    .map(format_duration)
                    .unwrap_or_else(|| "-".into());

                println!(
                    "{:<28} {:<24} {:<10} {:>10} {:>10} {}",
                    exp.id, exp.name, exp.status, loss, acc, duration
                );
            }
        }
        ExpCommands::Compare { a, b } => {
            let exp_a = store
                .get(&a)?
                .ok_or_else(|| anyhow::anyhow!("experiment not found: {a}"))?;
            let exp_b = store
                .get(&b)?
                .ok_or_else(|| anyhow::anyhow!("experiment not found: {b}"))?;
            println!("{}", compare::compare(&exp_a, &exp_b));
        }
        ExpCommands::Show { id } => {
            let exp = store
                .get(&id)?
                .ok_or_else(|| anyhow::anyhow!("experiment not found: {id}"))?;

            println!("Experiment: {}", exp.id);
            println!("  Name:       {}", exp.name);
            println!("  Status:     {}", exp.status);
            println!("  Script:     {}", exp.script.as_deref().unwrap_or("-"));
            println!("  Git commit: {}", exp.git_commit.as_deref().unwrap_or("-"));
            println!(
                "  Created:    {}",
                exp.created_at.format("%Y-%m-%d %H:%M:%S UTC")
            );
            if let Some(f) = exp.finished_at {
                println!("  Finished:   {}", f.format("%Y-%m-%d %H:%M:%S UTC"));
            }
            if let Some(d) = exp.duration_secs {
                println!("  Duration:   {}", format_duration(d));
            }

            if !exp.hyperparams.is_empty() {
                println!("\n  Hyperparameters:");
                let mut sorted: Vec<_> = exp.hyperparams.iter().collect();
                sorted.sort_by_key(|(k, _)| (*k).clone());
                for (k, v) in sorted {
                    println!("    {k}: {v}");
                }
            }

            if !exp.metrics.is_empty() {
                println!("\n  Metrics:");
                let mut sorted: Vec<_> = exp.metrics.iter().collect();
                sorted.sort_by_key(|(k, _)| (*k).clone());
                for (k, v) in sorted {
                    println!("    {k}: {v:.6}");
                }
            }
        }
        ExpCommands::Delete { id } => {
            if store.delete(&id)? {
                println!("Deleted experiment: {id}");
            } else {
                println!("Experiment not found: {id}");
            }
        }
    }
    Ok(())
}

fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else if secs < 3600.0 {
        format!("{:.0}m {:.0}s", secs / 60.0, secs % 60.0)
    } else {
        format!("{:.0}h {:.0}m", secs / 3600.0, (secs % 3600.0) / 60.0)
    }
}
