// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::Result;
use std::fs;
use tracing::info;

/// Scaffold a new Zernel ML project.
pub async fn run(name: &str) -> Result<()> {
    info!(name, "initializing project");

    let project_dir = std::path::Path::new(name);
    fs::create_dir_all(project_dir)?;
    fs::create_dir_all(project_dir.join("data"))?;
    fs::create_dir_all(project_dir.join("models"))?;
    fs::create_dir_all(project_dir.join("configs"))?;
    fs::create_dir_all(project_dir.join("scripts"))?;

    // Create zernel.toml project config
    let config = format!(
        r#"[project]
name = "{name}"
version = "0.1.0"

[training]
framework = "pytorch"
gpus = "auto"

[tracking]
enabled = true
auto_log = true
"#
    );
    fs::write(project_dir.join("zernel.toml"), config)?;

    // Create a starter training script
    let train_py = r#"#!/usr/bin/env python3
"""Zernel project training script."""

import torch
import torch.nn as nn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Your model and training loop here
    pass

if __name__ == "__main__":
    main()
"#;
    fs::write(project_dir.join("train.py"), train_py)?;

    println!("Initialized Zernel project: {name}/");
    println!("  zernel.toml   — project configuration");
    println!("  train.py      — starter training script");
    println!("  data/         — dataset directory");
    println!("  models/       — model checkpoints");
    println!("  configs/      — training configs");
    println!();
    println!("Next: cd {name} && zernel run train.py");

    Ok(())
}
