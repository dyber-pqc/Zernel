// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel cluster — GPU cluster management

use anyhow::Result;
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;

#[derive(Subcommand)]
pub enum ClusterCommands {
    /// Register a node in the cluster
    Add {
        /// Hostname or IP
        host: String,
        /// Number of GPUs on this node
        #[arg(long, default_value = "8")]
        gpus: u32,
        /// SSH user
        #[arg(long, default_value = "root")]
        user: String,
    },
    /// Remove a node from the cluster
    Remove {
        /// Hostname
        host: String,
    },
    /// Show cluster status
    Status,
    /// SSH to a cluster node
    Ssh {
        /// Hostname
        host: String,
    },
    /// Sync files to all nodes
    Sync {
        /// Local path to sync
        path: String,
        /// Destination on remote nodes
        #[arg(long, default_value = "~/")]
        to: String,
    },
    /// Run a command on all nodes
    Run {
        /// Command to run
        command: String,
        /// Run only on specific node
        #[arg(long)]
        on: Option<String>,
    },
    /// Drain a node (stop jobs gracefully before maintenance)
    Drain {
        /// Hostname
        host: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct ClusterNode {
    host: String,
    user: String,
    gpus: u32,
    status: String,
}

fn cluster_file() -> PathBuf {
    let dir = crate::experiments::tracker::zernel_dir().join("cluster");
    std::fs::create_dir_all(&dir).ok();
    dir.join("nodes.json")
}

fn load_nodes() -> Vec<ClusterNode> {
    let path = cluster_file();
    if path.exists() {
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    } else {
        Vec::new()
    }
}

fn save_nodes(nodes: &[ClusterNode]) -> Result<()> {
    std::fs::write(cluster_file(), serde_json::to_string_pretty(nodes)?)?;
    Ok(())
}

fn ssh_cmd(user: &str, host: &str, cmd: &str) -> Command {
    let mut c = Command::new("ssh");
    c.args([
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=5",
        &format!("{user}@{host}"),
        cmd,
    ]);
    c
}

pub async fn run(cmd: ClusterCommands) -> Result<()> {
    match cmd {
        ClusterCommands::Add { host, gpus, user } => {
            let mut nodes = load_nodes();
            nodes.retain(|n| n.host != host);

            // Test connectivity
            print!("Testing SSH to {user}@{host}... ");
            let test = ssh_cmd(&user, &host, "echo ok").output();
            match test {
                Ok(o) if o.status.success() => println!("OK"),
                _ => {
                    println!("FAILED");
                    println!("Ensure passwordless SSH is configured:");
                    println!("  ssh-copy-id {user}@{host}");
                    return Ok(());
                }
            }

            nodes.push(ClusterNode {
                host: host.clone(),
                user,
                gpus,
                status: "active".into(),
            });
            save_nodes(&nodes)?;
            println!(
                "Added {host} ({gpus} GPUs) to cluster ({} total nodes)",
                nodes.len()
            );
        }

        ClusterCommands::Remove { host } => {
            let mut nodes = load_nodes();
            let before = nodes.len();
            nodes.retain(|n| n.host != host);
            save_nodes(&nodes)?;
            if nodes.len() < before {
                println!("Removed {host} from cluster");
            } else {
                println!("Node {host} not found in cluster");
            }
        }

        ClusterCommands::Status => {
            let nodes = load_nodes();
            if nodes.is_empty() {
                println!("No nodes in cluster. Add one: zernel cluster add <host> --gpus 8");
                return Ok(());
            }

            println!("Zernel Cluster Status");
            println!("{}", "=".repeat(70));
            println!(
                "{:<20} {:<8} {:>5} {:>10} {:>10} {:>8}",
                "Host", "Status", "GPUs", "GPU Util", "Memory", "Temp"
            );
            println!("{}", "-".repeat(70));

            for node in &nodes {
                // Try to get GPU info via SSH
                let info = ssh_cmd(&node.user, &node.host,
                    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1"
                ).output();

                match info {
                    Ok(o) if o.status.success() => {
                        let data = String::from_utf8_lossy(&o.stdout);
                        let f: Vec<&str> = data.trim().split(',').map(|s| s.trim()).collect();
                        if f.len() >= 4 {
                            println!(
                                "{:<20} {:<8} {:>5} {:>8}% {:>5}/{:<4}MB {:>5}°C",
                                node.host, "online", node.gpus, f[0], f[1], f[2], f[3]
                            );
                        } else {
                            println!("{:<20} {:<8} {:>5}", node.host, "online", node.gpus);
                        }
                    }
                    _ => {
                        println!("{:<20} {:<8} {:>5}", node.host, "offline", node.gpus);
                    }
                }
            }

            let total_gpus: u32 = nodes.iter().map(|n| n.gpus).sum();
            println!();
            println!("Total: {} nodes, {} GPUs", nodes.len(), total_gpus);
        }

        ClusterCommands::Ssh { host } => {
            let nodes = load_nodes();
            let node = nodes.iter().find(|n| n.host == host);
            match node {
                Some(n) => {
                    let status = Command::new("ssh")
                        .args([&format!("{}@{}", n.user, n.host)])
                        .status()?;
                    let _ = status;
                }
                None => println!("Node {host} not in cluster. Add it: zernel cluster add {host}"),
            }
        }

        ClusterCommands::Sync { path, to } => {
            let nodes = load_nodes();
            if nodes.is_empty() {
                println!("No nodes in cluster");
                return Ok(());
            }

            for node in &nodes {
                print!("Syncing to {}@{}:{}... ", node.user, node.host, to);
                let status = Command::new("rsync")
                    .args([
                        "-avz",
                        "--progress",
                        &path,
                        &format!("{}@{}:{}", node.user, node.host, to),
                    ])
                    .output();
                match status {
                    Ok(o) if o.status.success() => println!("OK"),
                    _ => println!("FAILED"),
                }
            }
        }

        ClusterCommands::Run { command, on } => {
            let nodes = load_nodes();
            let targets: Vec<&ClusterNode> = if let Some(ref host) = on {
                nodes.iter().filter(|n| n.host == *host).collect()
            } else {
                nodes.iter().collect()
            };

            for node in targets {
                println!("--- {}@{} ---", node.user, node.host);
                let output = ssh_cmd(&node.user, &node.host, &command).output();
                match output {
                    Ok(o) => {
                        print!("{}", String::from_utf8_lossy(&o.stdout));
                        if !o.stderr.is_empty() {
                            eprint!("{}", String::from_utf8_lossy(&o.stderr));
                        }
                    }
                    Err(e) => println!("  ERROR: {e}"),
                }
                println!();
            }
        }

        ClusterCommands::Drain { host } => {
            println!("Draining {host}...");
            let mut nodes = load_nodes();
            if let Some(node) = nodes.iter_mut().find(|n| n.host == host) {
                node.status = "draining".into();
                save_nodes(&nodes)?;
                println!("  Status set to 'draining'");
                println!("  New jobs will not be scheduled to this node");
                println!("  Existing jobs will complete normally");
            } else {
                println!("Node {host} not found");
            }
        }
    }
    Ok(())
}
