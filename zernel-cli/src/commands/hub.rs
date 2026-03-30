// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel hub — Private model & dataset hub

use anyhow::Result;
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum HubCommands {
    /// Push a model or dataset to the hub
    Push {
        /// Local path to model/dataset
        path: String,
        /// Hub name (org/name)
        #[arg(long)]
        name: String,
        /// Version tag
        #[arg(long, default_value = "latest")]
        tag: String,
    },
    /// Pull a model or dataset from the hub
    Pull {
        /// Hub name (org/name:tag)
        name: String,
        /// Destination path
        #[arg(long, default_value = ".")]
        to: String,
    },
    /// List all items in the hub
    List,
    /// Search the hub
    Search {
        /// Search query
        query: String,
    },
    /// Start a local hub server
    Serve {
        /// Port
        #[arg(long, default_value = "9999")]
        port: u16,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct HubEntry {
    name: String,
    tag: String,
    path: String,
    size_bytes: u64,
    pushed_at: String,
}

fn hub_dir() -> PathBuf {
    let dir = crate::experiments::tracker::zernel_dir().join("hub");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn hub_registry() -> PathBuf {
    hub_dir().join("registry.json")
}

fn load_hub() -> Vec<HubEntry> {
    hub_registry()
        .exists()
        .then(|| {
            std::fs::read_to_string(hub_registry())
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        })
        .flatten()
        .unwrap_or_default()
}

fn save_hub(entries: &[HubEntry]) -> Result<()> {
    std::fs::write(hub_registry(), serde_json::to_string_pretty(entries)?)?;
    Ok(())
}

fn dir_size(path: &std::path::Path) -> u64 {
    if path.is_file() {
        return path.metadata().map(|m| m.len()).unwrap_or(0);
    }
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let ft = entry.file_type().ok();
            if ft.as_ref().map(|t| t.is_file()).unwrap_or(false) {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            } else if ft.as_ref().map(|t| t.is_dir()).unwrap_or(false) {
                total += dir_size(&entry.path());
            }
        }
    }
    total
}

pub async fn run(cmd: HubCommands) -> Result<()> {
    match cmd {
        HubCommands::Push { path, name, tag } => {
            let source = std::path::Path::new(&path);
            if !source.exists() {
                anyhow::bail!("path not found: {path}");
            }

            let dest = hub_dir().join(&name).join(&tag);
            std::fs::create_dir_all(&dest)?;

            println!("Pushing {path} → hub/{name}:{tag}");

            // Copy
            if source.is_file() {
                let fname = source.file_name().unwrap_or_default();
                std::fs::copy(source, dest.join(fname))?;
            } else {
                let status = std::process::Command::new("cp")
                    .args(["-r", &path, &dest.to_string_lossy()])
                    .status()?;
                let _ = status;
            }

            let size = dir_size(&dest);

            let mut hub = load_hub();
            hub.retain(|e| !(e.name == name && e.tag == tag));
            hub.push(HubEntry {
                name: name.clone(),
                tag: tag.clone(),
                path: dest.to_string_lossy().to_string(),
                size_bytes: size,
                pushed_at: chrono::Utc::now().to_rfc3339(),
            });
            save_hub(&hub)?;

            println!(
                "Pushed: {name}:{tag} ({:.1} MB)",
                size as f64 / (1024.0 * 1024.0)
            );
        }

        HubCommands::Pull { name, to } => {
            let (hub_name, tag) = name.split_once(':').unwrap_or((&name, "latest"));
            let hub = load_hub();
            let entry = hub
                .iter()
                .find(|e| e.name == hub_name && e.tag == tag)
                .ok_or_else(|| anyhow::anyhow!("not found in hub: {hub_name}:{tag}"))?;

            println!("Pulling {hub_name}:{tag} → {to}");
            let status = std::process::Command::new("cp")
                .args(["-r", &entry.path, &to])
                .status()?;

            if status.success() {
                println!("Done.");
            }
        }

        HubCommands::List => {
            let hub = load_hub();
            if hub.is_empty() {
                println!(
                    "Hub is empty. Push something: zernel hub push ./model --name my-org/model"
                );
                return Ok(());
            }

            let hdr = format!(
                "{:<30} {:<10} {:>10} {:>10}",
                "Name", "Tag", "Size", "Pushed"
            );
            println!("{hdr}");
            println!("{}", "-".repeat(65));
            for e in &hub {
                println!(
                    "{:<30} {:<10} {:>8.1} MB {}",
                    e.name,
                    e.tag,
                    e.size_bytes as f64 / (1024.0 * 1024.0),
                    &e.pushed_at[..10]
                );
            }
        }

        HubCommands::Search { query } => {
            let hub = load_hub();
            let q = query.to_lowercase();
            let results: Vec<_> = hub
                .iter()
                .filter(|e| e.name.to_lowercase().contains(&q))
                .collect();

            if results.is_empty() {
                println!("No results for '{query}'");
            } else {
                for e in results {
                    println!(
                        "{}:{} ({:.1} MB)",
                        e.name,
                        e.tag,
                        e.size_bytes as f64 / (1024.0 * 1024.0)
                    );
                }
            }
        }

        HubCommands::Serve { port } => {
            println!("Starting Zernel Hub server on port {port}...");
            println!("URL: http://0.0.0.0:{port}");
            println!("(Hub server coming in future release — for now, use zernel hub push/pull)");
        }
    }
    Ok(())
}
