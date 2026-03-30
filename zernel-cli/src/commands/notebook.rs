// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel notebook — Terminal notebook

use anyhow::{Context, Result};
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum NotebookCommands {
    /// Start Jupyter Lab
    Start {
        /// Port
        #[arg(long, default_value = "8888")]
        port: u16,
        /// Don't open browser
        #[arg(long)]
        no_browser: bool,
    },
    /// Open an existing notebook
    Open {
        /// Path to .ipynb file
        path: String,
    },
    /// Convert notebook to Python script
    Convert {
        /// Input .ipynb file
        input: String,
        /// Output format (py, html, pdf, md)
        #[arg(long, default_value = "py")]
        to: String,
    },
    /// List running notebook servers
    List,
    /// Stop a notebook server
    Stop,
}

pub async fn run(cmd: NotebookCommands) -> Result<()> {
    match cmd {
        NotebookCommands::Start { port, no_browser } => {
            println!("Starting Jupyter Lab on port {port}...");

            let mut args = vec![
                "lab".to_string(),
                format!("--port={port}"),
                "--ip=0.0.0.0".into(),
            ];
            if no_browser {
                args.push("--no-browser".into());
            }

            let status = tokio::process::Command::new("jupyter")
                .args(&args)
                .status()
                .await
                .with_context(|| "jupyter not found — install with: zernel install jupyter")?;

            if !status.success() {
                anyhow::bail!("Jupyter exited with code {}", status.code().unwrap_or(-1));
            }
        }

        NotebookCommands::Open { path } => {
            println!("Opening {path}...");
            let status = tokio::process::Command::new("jupyter")
                .args(["lab", &path])
                .status()
                .await
                .with_context(|| "jupyter not found")?;
            let _ = status;
        }

        NotebookCommands::Convert { input, to } => {
            let output = input.replace(".ipynb", &format!(".{to}"));
            println!("Converting {input} → {output}");

            let to_format = match to.as_str() {
                "py" => "script",
                "html" => "html",
                "pdf" => "pdf",
                "md" => "markdown",
                other => other,
            };

            let status = Command::new("jupyter")
                .args(["nbconvert", "--to", to_format, &input])
                .status()
                .with_context(|| "jupyter nbconvert not found")?;

            if status.success() {
                println!("Converted: {output}");
            } else {
                println!("Conversion failed.");
            }
        }

        NotebookCommands::List => {
            println!("Running Jupyter Servers:");
            let output = Command::new("jupyter")
                .args(["lab", "list"])
                .output()
                .with_context(|| "jupyter not found")?;
            print!("{}", String::from_utf8_lossy(&output.stdout));
        }

        NotebookCommands::Stop => {
            println!("Stopping Jupyter servers...");
            let status = Command::new("jupyter")
                .args(["lab", "stop"])
                .status()
                .with_context(|| "jupyter not found")?;
            if status.success() {
                println!("All servers stopped.");
            }
        }
    }
    Ok(())
}
