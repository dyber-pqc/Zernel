// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Subcommand)]
pub enum ModelCommands {
    /// Save a model checkpoint to the registry
    Save {
        /// Path to the checkpoint directory or file
        path: String,
        /// Model name
        #[arg(long)]
        name: Option<String>,
        /// Tag (e.g., production, staging)
        #[arg(long)]
        tag: Option<String>,
    },
    /// List models in the registry
    List,
    /// Deploy a model for inference
    Deploy {
        /// Model name:tag
        model: String,
        /// Deployment target
        #[arg(long, default_value = "local")]
        target: String,
        /// Port for inference server
        #[arg(long, default_value = "8080")]
        port: u16,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelEntry {
    name: String,
    version: String,
    tag: String,
    source_path: String,
    saved_at: String,
    git_commit: Option<String>,
    size_bytes: u64,
}

fn registry_dir() -> PathBuf {
    let dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".zernel")
        .join("models");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn registry_file() -> PathBuf {
    registry_dir().join("registry.json")
}

fn load_registry() -> Vec<ModelEntry> {
    let path = registry_file();
    if path.exists() {
        let data = std::fs::read_to_string(&path).unwrap_or_default();
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        Vec::new()
    }
}

fn save_registry(entries: &[ModelEntry]) -> Result<()> {
    let data = serde_json::to_string_pretty(entries)?;
    std::fs::write(registry_file(), data)?;
    Ok(())
}

fn dir_size(path: &Path) -> u64 {
    if path.is_file() {
        return path.metadata().map(|m| m.len()).unwrap_or(0);
    }
    walkdir(path)
}

fn walkdir(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let ft = entry.file_type().unwrap_or_else(|_| unreachable!());
            if ft.is_file() {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            } else if ft.is_dir() {
                total += walkdir(&entry.path());
            }
        }
    }
    total
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

pub async fn run(cmd: ModelCommands) -> Result<()> {
    match cmd {
        ModelCommands::Save { path, name, tag } => {
            let source = Path::new(&path);
            if !source.exists() {
                anyhow::bail!("path does not exist: {path}");
            }

            let model_name = name.unwrap_or_else(|| {
                source
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unnamed".into())
            });
            let model_tag = tag.unwrap_or_else(|| "latest".into());

            let git_commit = std::process::Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                    } else {
                        None
                    }
                });

            let size = dir_size(source);

            // Copy checkpoint to registry
            let dest = registry_dir().join(&model_name).join(&model_tag);
            std::fs::create_dir_all(&dest)?;

            if source.is_file() {
                let fname = source.file_name().unwrap();
                std::fs::copy(source, dest.join(fname))
                    .with_context(|| format!("failed to copy {path}"))?;
            } else {
                copy_dir_recursive(source, &dest)?;
            }

            let mut registry = load_registry();
            // Remove old entry with same name:tag
            registry.retain(|e| !(e.name == model_name && e.tag == model_tag));

            let version = format!(
                "{}.0.0",
                registry.iter().filter(|e| e.name == model_name).count() + 1
            );

            registry.push(ModelEntry {
                name: model_name.clone(),
                version: version.clone(),
                tag: model_tag.clone(),
                source_path: path.clone(),
                saved_at: Utc::now().to_rfc3339(),
                git_commit,
                size_bytes: size,
            });

            save_registry(&registry)?;

            println!("Saved model: {model_name}:{model_tag}");
            println!("  Version: {version}");
            println!("  Size:    {}", format_size(size));
            println!("  Path:    {}", dest.display());
        }
        ModelCommands::List => {
            let registry = load_registry();
            if registry.is_empty() {
                println!("No models saved. Use `zernel model save <path>` to register one.");
                return Ok(());
            }

            let header = format!(
                "{:<20} {:<10} {:<12} {:>10} {:>10}",
                "Name", "Version", "Tag", "Size", "Saved"
            );
            println!("{header}");
            println!("{}", "-".repeat(70));

            for entry in &registry {
                let saved = &entry.saved_at[..10]; // just date
                println!(
                    "{:<20} {:<10} {:<12} {:>10} {}",
                    entry.name,
                    entry.version,
                    entry.tag,
                    format_size(entry.size_bytes),
                    saved,
                );
            }
        }
        ModelCommands::Deploy {
            model,
            target,
            port,
        } => {
            let (name, tag) = model.split_once(':').unwrap_or((&model, "latest"));

            let registry = load_registry();
            let entry = registry
                .iter()
                .find(|e| e.name == name && e.tag == tag)
                .ok_or_else(|| anyhow::anyhow!("model not found: {name}:{tag}"))?;

            println!(
                "Deploying {name}:{tag} (v{}) to {target} on port {port}",
                entry.version
            );
            println!("  Source: {}", entry.source_path);
            println!();
            println!("(inference server deployment not yet implemented)");
            println!(
                "  For now, use: vllm serve {} --port {port}",
                entry.source_path
            );
        }
    }
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if ft.is_file() {
            std::fs::copy(entry.path(), &dest_path)?;
        } else if ft.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        }
    }
    Ok(())
}
