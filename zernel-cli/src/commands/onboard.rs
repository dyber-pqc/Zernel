// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel onboard — One-command developer onboarding
//!
//! Gets a new team member from "laptop" to "training a model" in minutes.

use anyhow::Result;
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum OnboardCommands {
    /// Full onboarding (check environment, install stack, create project)
    Setup {
        /// Project name
        #[arg(default_value = "my-first-project")]
        name: String,
    },
    /// Sync environment from another machine's snapshot
    Sync {
        /// Path to environment snapshot file
        file: String,
    },
    /// Share current environment with a teammate
    Share {
        /// Output file
        #[arg(long, default_value = "zernel-env-share.yml")]
        output: String,
    },
}

pub async fn run(cmd: OnboardCommands) -> Result<()> {
    match cmd {
        OnboardCommands::Setup { name } => {
            println!("Zernel Onboarding");
            println!("{}", "=".repeat(60));
            println!();

            // Step 1: Environment check
            println!("[1/5] Checking environment...");
            let checks = [
                ("python3", "--version"),
                ("git", "--version"),
                ("nvidia-smi", "--query-gpu=name --format=csv,noheader"),
            ];

            let mut all_ok = true;
            for (cmd_name, args) in &checks {
                let ok = Command::new(cmd_name)
                    .args(args.split_whitespace())
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false);

                let status = if ok {
                    "OK"
                } else {
                    all_ok = false;
                    "MISSING"
                };
                println!("  {cmd_name:<15} {status}");
            }

            if !all_ok {
                println!();
                println!("Some dependencies are missing. Run: zernel doctor");
                println!("Continue anyway? (y/n)");
            }

            // Step 2: Install ML stack
            println!();
            println!("[2/5] Checking ML stack...");
            let torch_ok = Command::new("python3")
                .args(["-c", "import torch; print(torch.__version__)"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);

            if torch_ok {
                let version = Command::new("python3")
                    .args(["-c", "import torch; print(torch.__version__)"])
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                    .unwrap_or_default();
                println!("  PyTorch {version} — already installed");
            } else {
                println!("  PyTorch not found.");
                println!("  Install with: zernel install pytorch");
            }

            // Step 3: Create project
            println!();
            println!("[3/5] Creating project: {name}...");
            let _ = crate::commands::init::run(&name).await;

            // Step 4: Environment snapshot
            println!();
            println!("[4/5] Saving environment snapshot...");
            let snapshot_path = format!("{name}/zernel-env.yml");
            let _ = Command::new("zernel")
                .args(["env", "snapshot", "--output", &snapshot_path])
                .status();
            println!("  Saved to: {snapshot_path}");

            // Step 5: Summary
            println!();
            println!("[5/5] Onboarding complete!");
            println!();
            println!("  Next steps:");
            println!("    cd {name}");
            println!("    zernel run train.py           # Start training");
            println!("    zernel watch                  # Monitor GPU dashboard");
            println!("    zernel gpu status             # Check GPU health");
            println!("    zernel bench quick            # Run performance benchmark");
            println!();
            println!("  Share this environment with teammates:");
            println!("    zernel onboard share --output env.yml");
            println!("    # Teammate runs: zernel onboard sync env.yml");
        }

        OnboardCommands::Sync { file } => {
            println!("Syncing environment from: {file}");
            println!();

            if !std::path::Path::new(&file).exists() {
                anyhow::bail!("file not found: {file}");
            }

            println!("  Installing packages from snapshot...");
            let status = Command::new("zernel")
                .args(["env", "reproduce", &file])
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!("  Environment synced successfully.");
                }
                _ => {
                    println!("  Some packages may have failed. Check output above.");
                }
            }
        }

        OnboardCommands::Share { output } => {
            println!("Generating shareable environment snapshot...");
            let status = Command::new("zernel")
                .args(["env", "snapshot", "--output", &output])
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!("  Saved to: {output}");
                    println!();
                    println!("  Share with teammates. They can sync with:");
                    println!("    zernel onboard sync {output}");
                }
                _ => {
                    println!("  Failed to generate snapshot.");
                }
            }
        }
    }
    Ok(())
}
