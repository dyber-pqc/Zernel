// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! SSH-based multi-node distributed training backend.
//!
//! Distributes training across multiple nodes via passwordless SSH.
//! Each node runs torchrun with the correct --node_rank and --master_addr.

use anyhow::{Context, Result};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Parse a hosts specification into a list of hostnames.
/// Accepts: "host1,host2,host3" or a path to a file with one host per line.
pub fn parse_hosts(spec: &str) -> Result<Vec<String>> {
    let path = std::path::Path::new(spec);
    if path.exists() {
        let content = std::fs::read_to_string(path)?;
        Ok(content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect())
    } else {
        Ok(spec.split(',').map(|s| s.trim().to_string()).collect())
    }
}

/// Launch a distributed training job across multiple nodes via SSH.
#[allow(clippy::too_many_arguments)]
pub async fn run_ssh_job(
    job_id: &str,
    script: &str,
    hosts: &[String],
    gpus_per_node: u32,
    framework: &str,
    backend: &str,
    args: &[String],
    log_dir: &std::path::Path,
) -> Result<i32> {
    let master_addr = &hosts[0];
    let master_port = 29500;
    let num_nodes = hosts.len();

    println!("SSH Multi-Node Launch");
    println!("  Master:  {master_addr}:{master_port}");
    println!("  Nodes:   {num_nodes}");
    println!("  Hosts:   {}", hosts.join(", "));
    println!();

    let mut handles = Vec::new();

    for (rank, host) in hosts.iter().enumerate() {
        let job_dir = format!("/tmp/zernel-{job_id}");

        // 1. Create remote working directory + copy script
        let setup = tokio::process::Command::new("ssh")
            .args([
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                host,
                &format!("mkdir -p {job_dir}"),
            ])
            .status()
            .await
            .with_context(|| format!("SSH to {host} failed — is passwordless SSH configured?"))?;

        if !setup.success() {
            anyhow::bail!("failed to create directory on {host}");
        }

        let scp = tokio::process::Command::new("scp")
            .args([
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                script,
                &format!("{host}:{job_dir}/"),
            ])
            .status()
            .await
            .with_context(|| format!("SCP to {host} failed"))?;

        if !scp.success() {
            anyhow::bail!("failed to copy script to {host}");
        }

        // 2. Build remote launch command
        let script_name = std::path::Path::new(script)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| script.to_string());

        let remote_cmd = match framework {
            "pytorch" => {
                let mut cmd_parts = vec![
                    "cd".to_string(),
                    job_dir.clone(),
                    "&&".into(),
                    "torchrun".into(),
                    format!("--nproc_per_node={gpus_per_node}"),
                    format!("--nnodes={num_nodes}"),
                    format!("--node_rank={rank}"),
                    format!("--master_addr={master_addr}"),
                    format!("--master_port={master_port}"),
                    script_name,
                ];
                cmd_parts.extend(args.iter().cloned());
                cmd_parts.join(" ")
            }
            "accelerate" | "hf" => {
                let total = gpus_per_node * num_nodes as u32;
                let mut cmd_parts = vec![
                    "cd".to_string(),
                    job_dir.clone(),
                    "&&".into(),
                    "accelerate".into(),
                    "launch".into(),
                    format!("--num_processes={total}"),
                    format!("--machine_rank={rank}"),
                    format!("--main_process_ip={master_addr}"),
                    format!("--main_process_port={master_port}"),
                    script_name,
                ];
                cmd_parts.extend(args.iter().cloned());
                cmd_parts.join(" ")
            }
            _ => anyhow::bail!("unsupported framework: {framework}"),
        };

        // Set NCCL environment
        let env_prefix = format!(
            "NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN{}",
            if backend == "nccl" {
                " NCCL_P2P_DISABLE=0"
            } else {
                ""
            }
        );

        let full_cmd = format!("{env_prefix} {remote_cmd}");

        // 3. Launch via SSH
        let host_clone = host.clone();
        let log_file = log_dir.join(format!("rank-{rank}.log"));

        let handle = tokio::spawn(async move {
            let mut child = tokio::process::Command::new("ssh")
                .args([
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=no",
                    &host_clone,
                    &full_cmd,
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("failed to spawn SSH");

            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            // Stream output with rank prefix
            let rank_clone = rank;
            let log_file_clone = log_file.clone();

            let out_handle = tokio::spawn(async move {
                if let Some(stdout) = stdout {
                    let mut reader = BufReader::new(stdout);
                    let mut line = String::new();
                    let mut log = std::fs::File::create(&log_file_clone).ok();
                    loop {
                        line.clear();
                        match reader.read_line(&mut line).await {
                            Ok(0) => break,
                            Ok(_) => {
                                print!("[rank {rank_clone}] {line}");
                                if let Some(ref mut f) = log {
                                    use std::io::Write;
                                    let _ = write!(f, "{line}");
                                }
                            }
                            Err(_) => break,
                        }
                    }
                }
            });

            let err_handle = tokio::spawn(async move {
                if let Some(stderr) = stderr {
                    let mut reader = BufReader::new(stderr);
                    let mut line = String::new();
                    loop {
                        line.clear();
                        match reader.read_line(&mut line).await {
                            Ok(0) => break,
                            Ok(_) => eprint!("[rank {rank}] {line}"),
                            Err(_) => break,
                        }
                    }
                }
            });

            let (_, _) = tokio::join!(out_handle, err_handle);
            let status = child
                .wait()
                .await
                .unwrap_or_else(|_| std::process::ExitStatus::default());
            status.code().unwrap_or(-1)
        });

        handles.push(handle);
    }

    // Wait for all nodes
    let mut exit_code = 0;
    for handle in handles {
        let code = handle.await.unwrap_or(-1);
        if code != 0 {
            exit_code = code;
        }
    }

    Ok(exit_code)
}

/// Cancel an SSH job by killing processes on all hosts.
pub async fn cancel_ssh_job(job_id: &str, hosts: &[String]) -> Result<()> {
    for host in hosts {
        println!("  Killing job on {host}...");
        let _ = tokio::process::Command::new("ssh")
            .args([
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                host,
                &format!("pkill -f zernel-{job_id} || true"),
            ])
            .status()
            .await;
    }
    Ok(())
}
