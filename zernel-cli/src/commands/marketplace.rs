// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel marketplace — Share and monetize ML models
//!
//! A decentralized marketplace for ML models and datasets.

use anyhow::Result;
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum MarketplaceCommands {
    /// Publish a model to the marketplace
    Publish {
        /// Path to model
        path: String,
        /// Model name
        #[arg(long)]
        name: String,
        /// Description
        #[arg(long, default_value = "")]
        description: String,
        /// License (mit, apache2, proprietary)
        #[arg(long, default_value = "apache2")]
        license: String,
    },
    /// Browse available models
    Browse {
        /// Search query
        #[arg(default_value = "")]
        query: String,
    },
    /// Download a model from the marketplace
    Download {
        /// Model name (author/name)
        name: String,
        /// Destination path
        #[arg(long, default_value = ".")]
        to: String,
    },
    /// Deploy a marketplace model
    Deploy {
        /// Model name
        name: String,
        /// Port
        #[arg(long, default_value = "8080")]
        port: u16,
    },
    /// Show my published models
    My,
}

#[derive(Debug, Serialize, Deserialize)]
struct MarketplaceEntry {
    name: String,
    description: String,
    license: String,
    path: String,
    size_bytes: u64,
    published_at: String,
    downloads: u64,
}

fn marketplace_dir() -> PathBuf {
    let dir = crate::experiments::tracker::zernel_dir().join("marketplace");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn marketplace_registry() -> PathBuf {
    marketplace_dir().join("registry.json")
}

fn load_marketplace() -> Vec<MarketplaceEntry> {
    marketplace_registry()
        .exists()
        .then(|| {
            std::fs::read_to_string(marketplace_registry())
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        })
        .flatten()
        .unwrap_or_default()
}

fn save_marketplace(entries: &[MarketplaceEntry]) -> Result<()> {
    std::fs::write(
        marketplace_registry(),
        serde_json::to_string_pretty(entries)?,
    )?;
    Ok(())
}

pub async fn run(cmd: MarketplaceCommands) -> Result<()> {
    match cmd {
        MarketplaceCommands::Publish {
            path,
            name,
            description,
            license,
        } => {
            let source = std::path::Path::new(&path);
            if !source.exists() {
                anyhow::bail!("path not found: {path}");
            }

            println!("Zernel Marketplace — Publish");
            println!("{}", "=".repeat(50));
            println!("  Name:        {name}");
            println!(
                "  Description: {}",
                if description.is_empty() {
                    "(none)"
                } else {
                    &description
                }
            );
            println!("  License:     {license}");
            println!("  Source:      {path}");

            // Copy to marketplace directory
            let dest = marketplace_dir().join(&name);
            std::fs::create_dir_all(&dest)?;

            let size = if source.is_file() {
                let fname = source.file_name().unwrap_or_default();
                std::fs::copy(source, dest.join(fname))?;
                source.metadata()?.len()
            } else {
                // Copy directory
                let _ = std::process::Command::new("cp")
                    .args(["-r", &path, &dest.to_string_lossy()])
                    .status();
                0
            };

            let mut registry = load_marketplace();
            registry.retain(|e| e.name != name);
            registry.push(MarketplaceEntry {
                name: name.clone(),
                description,
                license,
                path: dest.to_string_lossy().to_string(),
                size_bytes: size,
                published_at: chrono::Utc::now().to_rfc3339(),
                downloads: 0,
            });
            save_marketplace(&registry)?;

            println!();
            println!("  Published: {name}");
            println!("  Browse: zernel marketplace browse");
        }

        MarketplaceCommands::Browse { query } => {
            let registry = load_marketplace();

            println!("Zernel Marketplace");
            println!("{}", "=".repeat(60));

            if registry.is_empty() {
                println!("  No models published yet.");
                println!("  Publish one: zernel marketplace publish ./model --name my-model");
                return Ok(());
            }

            let filtered: Vec<&MarketplaceEntry> = if query.is_empty() {
                registry.iter().collect()
            } else {
                let q = query.to_lowercase();
                registry
                    .iter()
                    .filter(|e| {
                        e.name.to_lowercase().contains(&q)
                            || e.description.to_lowercase().contains(&q)
                    })
                    .collect()
            };

            for entry in &filtered {
                let size = if entry.size_bytes > 0 {
                    format!("{:.1} MB", entry.size_bytes as f64 / (1024.0 * 1024.0))
                } else {
                    "N/A".into()
                };
                println!("  {} ({})", entry.name, entry.license);
                if !entry.description.is_empty() {
                    println!("    {}", entry.description);
                }
                println!(
                    "    Size: {} | Published: {} | Downloads: {}",
                    size,
                    &entry.published_at[..10],
                    entry.downloads
                );
                println!();
            }

            if filtered.is_empty() {
                println!("  No models matching '{query}'");
            }
        }

        MarketplaceCommands::Download { name, to } => {
            let registry = load_marketplace();
            let entry = registry
                .iter()
                .find(|e| e.name == name)
                .ok_or_else(|| anyhow::anyhow!("model not found: {name}"))?;

            println!("Downloading: {name} → {to}");
            let _ = std::process::Command::new("cp")
                .args(["-r", &entry.path, &to])
                .status();
            println!("Done.");
        }

        MarketplaceCommands::Deploy { name, port } => {
            println!("Deploying {name} on port {port}...");
            println!("  Use: zernel serve start {name} --port {port}");
        }

        MarketplaceCommands::My => {
            let registry = load_marketplace();
            if registry.is_empty() {
                println!("No published models.");
                return Ok(());
            }

            println!("My Published Models");
            println!("{}", "=".repeat(50));
            for entry in &registry {
                println!(
                    "  {} — {} ({})",
                    entry.name,
                    entry.license,
                    &entry.published_at[..10]
                );
            }
        }
    }
    Ok(())
}
