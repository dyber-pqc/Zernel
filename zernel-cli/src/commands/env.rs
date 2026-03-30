// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel env — Environment management

use anyhow::Result;
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum EnvCommands {
    /// Snapshot current environment (Python, CUDA, driver, packages)
    Snapshot {
        /// Output file
        #[arg(long, default_value = "zernel-env.yml")]
        output: String,
    },
    /// Diff two environment snapshots
    Diff {
        /// First snapshot
        a: String,
        /// Second snapshot
        b: String,
    },
    /// Reproduce an environment from a snapshot
    Reproduce {
        /// Snapshot file to reproduce
        file: String,
    },
    /// Export environment as Dockerfile
    Export {
        /// Export format (docker, conda, pip)
        #[arg(long, default_value = "docker")]
        format: String,
        /// Output file
        #[arg(long, default_value = "Dockerfile.zernel")]
        output: String,
    },
    /// Show current environment
    Show,
}

fn run_cmd(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "N/A".into())
}

pub async fn run(cmd: EnvCommands) -> Result<()> {
    match cmd {
        EnvCommands::Show | EnvCommands::Snapshot { .. } => {
            let os_info = run_cmd("uname", &["-srm"]);
            let python = run_cmd("python3", &["--version"]);
            let pip_list = run_cmd("pip", &["list", "--format=freeze"]);
            let cuda = run_cmd("nvcc", &["--version"]);
            let driver = run_cmd(
                "nvidia-smi",
                &["--query-gpu=driver_version", "--format=csv,noheader"],
            );
            let gpu_name = run_cmd("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]);
            let torch_ver = run_cmd("python3", &["-c", "import torch; print(torch.__version__)"]);
            let torch_cuda = run_cmd(
                "python3",
                &["-c", "import torch; print(torch.version.cuda)"],
            );

            let snapshot = format!(
                "# Zernel Environment Snapshot\n\
                 # Generated: {}\n\n\
                 os: {}\n\
                 python: {}\n\
                 nvidia_driver: {}\n\
                 gpu: {}\n\
                 cuda_toolkit: {}\n\
                 torch: {}\n\
                 torch_cuda: {}\n\n\
                 # pip packages\n\
                 packages:\n{}\n",
                chrono::Utc::now().to_rfc3339(),
                os_info,
                python,
                driver.lines().next().unwrap_or("N/A"),
                gpu_name.lines().next().unwrap_or("N/A"),
                cuda.lines().last().unwrap_or("N/A"),
                torch_ver,
                torch_cuda,
                pip_list
                    .lines()
                    .map(|l| format!("  - {l}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );

            if let EnvCommands::Snapshot { output } = cmd {
                std::fs::write(&output, &snapshot)?;
                println!("Environment snapshot saved to: {output}");
            } else {
                print!("{snapshot}");
            }
        }

        EnvCommands::Diff { a, b } => {
            let a_content = std::fs::read_to_string(&a)?;
            let b_content = std::fs::read_to_string(&b)?;

            println!("Environment Diff: {a} vs {b}");
            println!("{}", "=".repeat(60));

            let a_lines: std::collections::HashSet<&str> = a_content.lines().collect();
            let b_lines: std::collections::HashSet<&str> = b_content.lines().collect();

            let only_a: Vec<&&str> = a_lines.difference(&b_lines).collect();
            let only_b: Vec<&&str> = b_lines.difference(&a_lines).collect();

            if !only_a.is_empty() {
                println!("\nOnly in {a}:");
                for line in &only_a {
                    if !line.starts_with('#') && !line.is_empty() {
                        println!("  - {line}");
                    }
                }
            }

            if !only_b.is_empty() {
                println!("\nOnly in {b}:");
                for line in &only_b {
                    if !line.starts_with('#') && !line.is_empty() {
                        println!("  + {line}");
                    }
                }
            }

            if only_a.is_empty() && only_b.is_empty() {
                println!("Environments are identical.");
            }
        }

        EnvCommands::Reproduce { file } => {
            let content = std::fs::read_to_string(&file)?;
            println!("Reproducing environment from: {file}");

            // Extract pip packages
            let mut in_packages = false;
            let mut packages = Vec::new();
            for line in content.lines() {
                if line.starts_with("packages:") {
                    in_packages = true;
                    continue;
                }
                if in_packages {
                    if let Some(pkg) = line.strip_prefix("  - ") {
                        packages.push(pkg.to_string());
                    } else {
                        in_packages = false;
                    }
                }
            }

            if packages.is_empty() {
                println!("No packages found in snapshot.");
                return Ok(());
            }

            println!("Installing {} packages...", packages.len());
            let reqs = packages.join("\n");
            let req_file = "/tmp/zernel-requirements.txt";
            std::fs::write(req_file, &reqs)?;

            let status = Command::new("pip")
                .args(["install", "-r", req_file])
                .status()?;

            if status.success() {
                println!("Environment reproduced successfully.");
            } else {
                println!("Some packages failed to install.");
            }
        }

        EnvCommands::Export { format, output } => match format.as_str() {
            "docker" => {
                let pip_list = run_cmd("pip", &["list", "--format=freeze"]);
                let dockerfile =
                        "# Generated by zernel env export\n\
                         FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04\n\n\
                         RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*\n\n\
                         COPY requirements.txt /tmp/\n\
                         RUN pip install --no-cache-dir -r /tmp/requirements.txt\n\n\
                         WORKDIR /workspace\n";
                std::fs::write(&output, dockerfile)?;
                std::fs::write("requirements.txt", &pip_list)?;
                println!("Exported to: {output} + requirements.txt");
                println!("Build: docker build -f {output} -t my-env .");
            }
            "pip" => {
                let pip_list = run_cmd("pip", &["list", "--format=freeze"]);
                std::fs::write(&output, &pip_list)?;
                println!("Exported to: {output}");
                println!("Reproduce: pip install -r {output}");
            }
            other => {
                println!("Unknown format: {other}. Available: docker, pip");
            }
        },
    }
    Ok(())
}
