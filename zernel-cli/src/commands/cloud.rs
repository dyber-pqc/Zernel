// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel cloud — One command GPU cluster management
//!
//! Launch, manage, and destroy GPU clusters across providers.

use anyhow::Result;
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum CloudCommands {
    /// Launch a GPU cluster
    Launch {
        /// Number of GPUs
        #[arg(long, default_value = "1")]
        gpus: u32,
        /// Cloud provider (lambda, aws, gcp, azure, vast)
        #[arg(long, default_value = "lambda")]
        provider: String,
        /// Instance type (provider-specific)
        #[arg(long)]
        instance: Option<String>,
        /// Region
        #[arg(long, default_value = "us-east-1")]
        region: String,
    },
    /// SSH into the running cluster
    Ssh,
    /// Show cluster status
    Status,
    /// Destroy the cluster
    Destroy {
        /// Skip confirmation
        #[arg(long)]
        yes: bool,
    },
    /// List available GPU instances and pricing
    Pricing {
        /// Provider to check
        #[arg(long, default_value = "all")]
        provider: String,
    },
}

fn cloud_state_file() -> std::path::PathBuf {
    crate::experiments::tracker::zernel_dir().join("cloud-state.json")
}

pub async fn run(cmd: CloudCommands) -> Result<()> {
    match cmd {
        CloudCommands::Launch {
            gpus,
            provider,
            instance,
            region,
        } => {
            println!("Zernel Cloud Launch");
            println!("{}", "=".repeat(50));
            println!("  Provider: {provider}");
            println!("  GPUs:     {gpus}");
            println!("  Region:   {region}");
            println!();

            match provider.as_str() {
                "lambda" => {
                    let instance_type = instance.unwrap_or_else(|| "gpu_1x_a100".into());
                    println!("  Instance: {instance_type}");
                    println!();
                    println!("Launching via Lambda Labs API...");
                    println!("  Requires: LAMBDA_API_KEY environment variable");
                    println!("  Get key at: https://cloud.lambdalabs.com/api-keys");
                    println!();

                    let api_key = std::env::var("LAMBDA_API_KEY").unwrap_or_default();
                    if api_key.is_empty() {
                        println!("  Set your API key:");
                        println!("    export LAMBDA_API_KEY=your-key-here");
                        println!("    zernel cloud launch --provider lambda --gpus {gpus}");
                        return Ok(());
                    }

                    let body = serde_json::json!({
                        "region_name": region,
                        "instance_type_name": instance_type,
                        "quantity": 1,
                    });

                    println!("  Sending launch request...");
                    let output = Command::new("curl")
                        .args([
                            "-s",
                            "-X",
                            "POST",
                            "https://cloud.lambdalabs.com/api/v1/instance-operations/launch",
                            "-H",
                            &format!("Authorization: Bearer {api_key}"),
                            "-H",
                            "Content-Type: application/json",
                            "-d",
                            &body.to_string(),
                        ])
                        .output();

                    match output {
                        Ok(o) => {
                            let resp = String::from_utf8_lossy(&o.stdout);
                            println!("  Response: {resp}");
                            // Save state
                            std::fs::write(cloud_state_file(), resp.to_string())?;
                        }
                        Err(e) => println!("  Error: {e}"),
                    }
                }

                "aws" => {
                    let instance_type = instance.unwrap_or_else(|| "p4d.24xlarge".into());
                    println!("  Instance: {instance_type}");
                    println!();
                    println!("  Run:");
                    println!("    aws ec2 run-instances \\");
                    println!("      --instance-type {instance_type} \\");
                    println!("      --image-id ami-zernel \\");
                    println!("      --region {region} \\");
                    println!("      --key-name your-key");
                }

                "vast" => {
                    println!("  Searching Vast.ai for {gpus}x GPU instances...");
                    println!();
                    println!("  Run:");
                    println!("    vastai search offers 'gpu_name=A100 num_gpus={gpus}'");
                    println!("    vastai create instance <offer_id> --image zernel/zernel:latest");
                }

                other => {
                    println!("  Provider '{other}' not yet supported.");
                    println!("  Available: lambda, aws, vast");
                }
            }
        }

        CloudCommands::Ssh => {
            let state_file = cloud_state_file();
            if state_file.exists() {
                let state = std::fs::read_to_string(&state_file)?;
                println!("Connecting to cluster...");
                println!("  State: {state}");
                println!("  Run: ssh ubuntu@<instance-ip>");
            } else {
                println!("No active cluster. Launch one: zernel cloud launch --gpus 8");
            }
        }

        CloudCommands::Status => {
            let state_file = cloud_state_file();
            if state_file.exists() {
                let state = std::fs::read_to_string(&state_file)?;
                println!("Zernel Cloud Status");
                println!("{}", "=".repeat(50));
                println!("{state}");
            } else {
                println!("No active cluster.");
            }
        }

        CloudCommands::Destroy { yes } => {
            if !yes {
                println!("This will destroy your GPU cluster. Add --yes to confirm.");
                return Ok(());
            }
            let state_file = cloud_state_file();
            if state_file.exists() {
                std::fs::remove_file(&state_file)?;
                println!("Cluster state cleared.");
                println!(
                    "Note: You may need to manually terminate instances in your cloud console."
                );
            } else {
                println!("No active cluster to destroy.");
            }
        }

        CloudCommands::Pricing { provider } => {
            println!("GPU Cloud Pricing Reference");
            println!("{}", "=".repeat(60));
            println!();
            println!(
                "{:<20} {:<15} {:>10} {:>12}",
                "Provider", "GPU", "Per Hour", "Per Month"
            );
            println!("{}", "-".repeat(60));

            let pricing = [
                ("Lambda Labs", "1x A100 80GB", "$1.10", "$792"),
                ("Lambda Labs", "8x A100 80GB", "$8.80", "$6,336"),
                ("AWS", "1x A100 (p4d)", "$3.67", "$2,642"),
                ("AWS", "8x A100 (p4d)", "$32.77", "$23,594"),
                ("Vast.ai", "1x A100 80GB", "$0.80-1.50", "$576-1,080"),
                ("Vast.ai", "1x RTX 4090", "$0.20-0.40", "$144-288"),
                ("Google Cloud", "1x A100 (a2)", "$3.67", "$2,642"),
                ("Azure", "1x A100 (NC)", "$3.67", "$2,642"),
                ("RunPod", "1x A100 80GB", "$1.64", "$1,181"),
                ("Hetzner", "1x RTX 3060", "$0.07", "$50"),
            ];

            for (prov, gpu, hourly, monthly) in &pricing {
                if provider == "all" || prov.to_lowercase().contains(&provider.to_lowercase()) {
                    println!("{:<20} {:<15} {:>10} {:>12}", prov, gpu, hourly, monthly);
                }
            }

            println!();
            println!("Prices are approximate and subject to change.");
            println!("Launch: zernel cloud launch --provider lambda --gpus 8");
        }
    }
    Ok(())
}
