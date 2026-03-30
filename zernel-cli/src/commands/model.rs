// Copyright (C) 2026 Dyber, Inc. — Proprietary

use crate::validation;
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
        /// Deployment target (local, docker, sagemaker)
        #[arg(long, default_value = "local")]
        target: String,
        /// Port for inference server
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Docker registry to push to (for --target docker)
        #[arg(long)]
        registry: Option<String>,
        /// AWS region (for --target sagemaker)
        #[arg(long, default_value = "us-east-1")]
        region: String,
        /// SageMaker instance type
        #[arg(long, default_value = "ml.g5.xlarge")]
        instance_type: String,
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
            let Ok(ft) = entry.file_type() else {
                continue;
            };
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

            // Validate name and tag to prevent path traversal
            validation::validate_name(&model_name)?;
            validation::validate_tag(&model_tag)?;

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
                let fname = source
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("source path has no filename"))?;
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
            registry: docker_registry,
            region,
            instance_type,
        } => {
            let (name, tag) = model.split_once(':').unwrap_or((&model, "latest"));

            let registry = load_registry();
            let entry = registry
                .iter()
                .find(|e| e.name == name && e.tag == tag)
                .ok_or_else(|| anyhow::anyhow!("model not found: {name}:{tag}"))?;

            let model_path = registry_dir().join(name).join(tag);

            println!(
                "Deploying {name}:{tag} (v{}) to {target} on port {port}",
                entry.version
            );
            println!("  Source: {}", model_path.display());

            match target.as_str() {
                "local" => {
                    // Check vllm is installed
                    let vllm_check = std::process::Command::new("python3")
                        .args(["-c", "import vllm; print(vllm.__version__)"])
                        .output();

                    match vllm_check {
                        Ok(output) if output.status.success() => {
                            let version =
                                String::from_utf8_lossy(&output.stdout).trim().to_string();
                            println!("  vLLM:   v{version}");
                        }
                        _ => {
                            println!();
                            println!("  vLLM not found. Install with: pip install vllm");
                            println!(
                                "  Or run manually: python3 -m vllm.entrypoints.openai.api_server --model {} --port {port}",
                                model_path.display()
                            );
                            return Ok(());
                        }
                    }

                    println!();
                    println!("Starting vLLM inference server...");
                    println!("  URL: http://localhost:{port}/v1");
                    println!("  Press Ctrl+C to stop");
                    println!();

                    // Launch vLLM
                    let status = tokio::process::Command::new("python3")
                        .args([
                            "-m",
                            "vllm.entrypoints.openai.api_server",
                            "--model",
                            &model_path.to_string_lossy(),
                            "--port",
                            &port.to_string(),
                        ])
                        .status()
                        .await
                        .with_context(|| "failed to start vLLM server")?;

                    if !status.success() {
                        anyhow::bail!("vLLM exited with code {}", status.code().unwrap_or(-1));
                    }
                }
                "docker" => {
                    println!();

                    // Generate Dockerfile
                    let dockerfile_content = format!(
                        "FROM vllm/vllm-openai:latest\n\
                         COPY . /model\n\
                         EXPOSE {port}\n\
                         ENTRYPOINT [\"python3\", \"-m\", \"vllm.entrypoints.openai.api_server\", \
                         \"--model\", \"/model\", \"--port\", \"{port}\"]\n"
                    );

                    let dockerfile_path = model_path.join("Dockerfile.zernel");
                    std::fs::write(&dockerfile_path, &dockerfile_content)?;
                    println!("  Generated: {}", dockerfile_path.display());

                    let image_tag = format!("zernel-{name}:{tag}");
                    println!("  Building Docker image: {image_tag}");

                    let build_status = std::process::Command::new("docker")
                        .args(["build", "-t", &image_tag, "-f"])
                        .arg(&dockerfile_path)
                        .arg(&model_path)
                        .status()?;

                    if !build_status.success() {
                        anyhow::bail!("docker build failed");
                    }

                    println!("  Image built: {image_tag}");

                    if let Some(ref reg) = docker_registry {
                        let remote_tag = format!("{reg}/{image_tag}");
                        println!("  Pushing to {remote_tag}...");

                        std::process::Command::new("docker")
                            .args(["tag", &image_tag, &remote_tag])
                            .status()?;

                        let push_status = std::process::Command::new("docker")
                            .args(["push", &remote_tag])
                            .status()?;

                        if push_status.success() {
                            println!("  Pushed: {remote_tag}");
                        } else {
                            anyhow::bail!("docker push failed");
                        }
                    }

                    println!();
                    println!("Run locally: docker run --gpus all -p {port}:{port} {image_tag}");
                }

                "sagemaker" => {
                    println!();
                    println!("  Region:   {region}");
                    println!("  Instance: {instance_type}");

                    // Check AWS CLI
                    let aws_check = std::process::Command::new("aws")
                        .args(["sts", "get-caller-identity"])
                        .output();

                    match aws_check {
                        Ok(output) if output.status.success() => {
                            let identity = String::from_utf8_lossy(&output.stdout);
                            println!("  AWS:      authenticated");
                            let _ = identity;
                        }
                        _ => {
                            println!();
                            println!("  AWS CLI not configured. Run: aws configure");
                            return Ok(());
                        }
                    }

                    let s3_path = format!("s3://zernel-models/{name}/{tag}/");
                    println!("  Uploading to {s3_path}...");

                    let sync_status = std::process::Command::new("aws")
                        .args([
                            "s3",
                            "sync",
                            &model_path.to_string_lossy(),
                            &s3_path,
                            "--region",
                            &region,
                        ])
                        .status()?;

                    if !sync_status.success() {
                        anyhow::bail!("aws s3 sync failed");
                    }

                    let sm_model_name = format!("zernel-{name}-{tag}");

                    println!("  Creating SageMaker model: {sm_model_name}");
                    let create_status = std::process::Command::new("aws")
                        .args([
                            "sagemaker", "create-model",
                            "--model-name", &sm_model_name,
                            "--primary-container",
                            &format!("Image=763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04,ModelDataUrl={s3_path}"),
                            "--execution-role-arn", "arn:aws:iam::role/SageMakerExecutionRole",
                            "--region", &region,
                        ])
                        .status()?;

                    if !create_status.success() {
                        println!("  SageMaker model creation failed.");
                        println!("  You may need to configure the IAM role and container image.");
                        println!(
                            "  Manual: aws sagemaker create-model --model-name {sm_model_name} ..."
                        );
                        return Ok(());
                    }

                    println!("  Creating endpoint config...");
                    let _ = std::process::Command::new("aws")
                        .args([
                            "sagemaker", "create-endpoint-config",
                            "--endpoint-config-name", &format!("{sm_model_name}-config"),
                            "--production-variants",
                            &format!("VariantName=default,ModelName={sm_model_name},InstanceType={instance_type},InitialInstanceCount=1"),
                            "--region", &region,
                        ])
                        .status();

                    println!("  Creating endpoint...");
                    let _ = std::process::Command::new("aws")
                        .args([
                            "sagemaker",
                            "create-endpoint",
                            "--endpoint-name",
                            &sm_model_name,
                            "--endpoint-config-name",
                            &format!("{sm_model_name}-config"),
                            "--region",
                            &region,
                        ])
                        .status();

                    println!();
                    println!("  Endpoint: {sm_model_name}");
                    println!("  Check status: aws sagemaker describe-endpoint --endpoint-name {sm_model_name} --region {region}");
                }

                other => {
                    println!();
                    println!("Unknown target: '{other}'");
                    println!("Available: local, docker, sagemaker");
                }
            }
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
