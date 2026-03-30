// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand)]
pub enum ExpCommands {
    /// List all experiments
    List,
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
    match cmd {
        ExpCommands::List => {
            // TODO: Query SQLite experiment store
            println!("No experiments yet. Run `zernel run <script>` to create one.");
        }
        ExpCommands::Compare { a, b } => {
            println!("Comparing experiments: {a} vs {b}");
            println!("(not yet implemented)");
        }
        ExpCommands::Show { id } => {
            println!("Experiment: {id}");
            println!("(not yet implemented)");
        }
        ExpCommands::Delete { id } => {
            println!("Deleting experiment: {id}");
            println!("(not yet implemented)");
        }
    }
    Ok(())
}
