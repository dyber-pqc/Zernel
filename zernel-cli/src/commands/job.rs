// Copyright (C) 2026 Dyber, Inc. — Proprietary

use crate::experiments::tracker;
use anyhow::{Context, Result};
use chrono::Utc;
use clap::Subcommand;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};

#[derive(Subcommand)]
pub enum JobCommands {
    /// Submit a distributed training job
    Submit {
        /// Script to run
        script: String,
        /// GPUs per node
        #[arg(long, default_value = "1")]
        gpus_per_node: u32,
        /// Number of nodes
        #[arg(long, default_value = "1")]
        nodes: u32,
        /// Framework (pytorch, jax)
        #[arg(long, default_value = "pytorch")]
        framework: String,
        /// Communication backend (nccl, gloo)
        #[arg(long, default_value = "nccl")]
        backend: String,
        /// Additional script arguments
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// List running and completed jobs
    List,
    /// Show job status and output
    Status {
        /// Job ID
        id: String,
    },
    /// Cancel a running job
    Cancel {
        /// Job ID
        id: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct Job {
    id: String,
    script: String,
    status: String,
    gpus_per_node: u32,
    nodes: u32,
    framework: String,
    backend: String,
    pid: Option<u32>,
    submitted_at: String,
    finished_at: Option<String>,
    exit_code: Option<i32>,
}

fn jobs_db_path() -> PathBuf {
    let dir = tracker::zernel_dir().join("jobs");
    std::fs::create_dir_all(&dir).ok();
    dir.join("jobs.db")
}

fn jobs_log_dir(job_id: &str) -> PathBuf {
    let dir = tracker::zernel_dir().join("jobs").join(job_id);
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn open_jobs_db() -> Result<Connection> {
    let conn = Connection::open(jobs_db_path())?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            script TEXT NOT NULL,
            status TEXT NOT NULL,
            gpus_per_node INTEGER NOT NULL,
            nodes INTEGER NOT NULL,
            framework TEXT NOT NULL,
            backend TEXT NOT NULL,
            pid INTEGER,
            submitted_at TEXT NOT NULL,
            finished_at TEXT,
            exit_code INTEGER
        );",
    )?;
    Ok(conn)
}

fn generate_job_id() -> String {
    let now = chrono::Utc::now();
    let short = &uuid::Uuid::new_v4().to_string()[..8];
    format!("job-{}-{}", now.format("%Y%m%d-%H%M%S"), short)
}

fn detect_gpu_count() -> u32 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=count", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .lines()
                    .next()
                    .and_then(|s| s.parse().ok())
            } else {
                None
            }
        })
        .unwrap_or(0)
}

pub async fn run(cmd: JobCommands) -> Result<()> {
    match cmd {
        JobCommands::Submit {
            script,
            gpus_per_node,
            nodes,
            framework,
            backend,
            args,
        } => {
            let script_path = std::path::Path::new(&script);
            if !script_path.exists() {
                anyhow::bail!("script not found: {script}");
            }

            let detected_gpus = detect_gpu_count();
            let gpus = if gpus_per_node == 1 && detected_gpus > 1 {
                println!("Detected {detected_gpus} GPUs. Using all.");
                detected_gpus
            } else {
                gpus_per_node
            };

            let total_procs = gpus * nodes;
            let job_id = generate_job_id();
            let log_dir = jobs_log_dir(&job_id);
            let log_path = log_dir.join("output.log");

            // Build launch command
            let (launcher, launch_args) = match framework.as_str() {
                "pytorch" => {
                    let mut a = vec![
                        format!("--nproc_per_node={gpus}"),
                        format!("--nnodes={nodes}"),
                        "--master_addr=localhost".into(),
                        "--master_port=29500".into(),
                        script.clone(),
                    ];
                    a.extend(args.clone());
                    ("torchrun".to_string(), a)
                }
                "accelerate" | "hf" => {
                    let mut a = vec![
                        "launch".into(),
                        format!("--num_processes={total_procs}"),
                        script.clone(),
                    ];
                    a.extend(args.clone());
                    ("accelerate".to_string(), a)
                }
                other => {
                    anyhow::bail!("unsupported framework: {other}. Use 'pytorch' or 'accelerate'.");
                }
            };

            println!("Zernel Job Submit");
            println!("  Job ID:      {job_id}");
            println!("  Script:      {script}");
            println!("  Framework:   {framework}");
            println!("  Backend:     {backend}");
            println!("  GPUs/node:   {gpus}");
            println!("  Nodes:       {nodes}");
            println!("  Total procs: {total_procs}");
            println!("  Launcher:    {launcher} {}", launch_args.join(" "));
            println!("  Log:         {}", log_path.display());
            println!();

            // Set environment
            let mut env_vars = vec![
                ("NCCL_SOCKET_IFNAME", "eth0,en0".to_string()),
                ("NCCL_DEBUG", "WARN".to_string()),
            ];
            if backend == "nccl" {
                env_vars.push(("NCCL_P2P_DISABLE", "0".to_string()));
            }

            // Spawn process
            let mut child = tokio::process::Command::new(&launcher)
                .args(&launch_args)
                .envs(env_vars)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .with_context(|| format!("failed to launch {launcher}. Is it installed?"))?;

            let pid = child.id().unwrap_or(0) as u32;

            // Record in DB
            let conn = open_jobs_db()?;
            conn.execute(
                "INSERT INTO jobs (id, script, status, gpus_per_node, nodes, framework, backend, pid, submitted_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                (
                    &job_id, &script, "running", gpus, nodes, &framework, &backend, pid,
                    Utc::now().to_rfc3339(),
                ),
            )?;

            println!("Job started (PID: {pid})");
            println!();

            // Capture output
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            let log_path_clone = log_path.clone();

            let stdout_handle = tokio::spawn(async move {
                let Some(stdout) = stdout else { return };
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                let mut log_file = std::fs::File::create(&log_path_clone).ok();

                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break,
                        Ok(_) => {
                            print!("{line}");
                            if let Some(ref mut f) = log_file {
                                use std::io::Write;
                                let _ = f.write_all(line.as_bytes());
                            }
                        }
                        Err(_) => break,
                    }
                }
            });

            let stderr_handle = tokio::spawn(async move {
                let Some(stderr) = stderr else { return };
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break,
                        Ok(_) => eprint!("{line}"),
                        Err(_) => break,
                    }
                }
            });

            let (_, _) = tokio::join!(stdout_handle, stderr_handle);
            let status = child.wait().await?;
            let exit_code = status.code().unwrap_or(-1);

            // Update DB
            let final_status = if status.success() { "done" } else { "failed" };
            conn.execute(
                "UPDATE jobs SET status = ?1, finished_at = ?2, exit_code = ?3 WHERE id = ?4",
                (final_status, Utc::now().to_rfc3339(), exit_code, &job_id),
            )?;

            println!();
            println!("---");
            println!("  Status: {final_status}");
            println!("  Exit:   {exit_code}");
            println!("  Job ID: {job_id}");
            println!("  Log:    {}", log_path.display());
        }

        JobCommands::List => {
            let conn = open_jobs_db()?;
            let mut stmt = conn.prepare(
                "SELECT id, script, status, gpus_per_node, nodes, framework, submitted_at, exit_code FROM jobs ORDER BY submitted_at DESC LIMIT 20",
            )?;

            let mut jobs = Vec::new();
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                jobs.push(Job {
                    id: row.get(0)?,
                    script: row.get(1)?,
                    status: row.get(2)?,
                    gpus_per_node: row.get(3)?,
                    nodes: row.get(4)?,
                    framework: row.get(5)?,
                    backend: String::new(),
                    pid: None,
                    submitted_at: row.get(6)?,
                    finished_at: None,
                    exit_code: row.get(7)?,
                });
            }

            if jobs.is_empty() {
                println!("No jobs. Submit one with: zernel job submit <script>");
                return Ok(());
            }

            let header = format!(
                "{:<30} {:<20} {:<10} {:>5} {:>5} {:<12} {:>6}",
                "ID", "Script", "Status", "GPUs", "Nodes", "Framework", "Exit"
            );
            println!("{header}");
            println!("{}", "-".repeat(95));

            for j in &jobs {
                let exit_str = j
                    .exit_code
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "-".into());
                println!(
                    "{:<30} {:<20} {:<10} {:>5} {:>5} {:<12} {:>6}",
                    j.id, j.script, j.status, j.gpus_per_node, j.nodes, j.framework, exit_str
                );
            }
        }

        JobCommands::Status { id } => {
            let conn = open_jobs_db()?;
            let mut stmt = conn.prepare(
                "SELECT id, script, status, gpus_per_node, nodes, framework, backend, pid, submitted_at, finished_at, exit_code FROM jobs WHERE id = ?1",
            )?;

            let job = stmt
                .query_row([&id], |row| {
                    Ok(Job {
                        id: row.get(0)?,
                        script: row.get(1)?,
                        status: row.get(2)?,
                        gpus_per_node: row.get(3)?,
                        nodes: row.get(4)?,
                        framework: row.get(5)?,
                        backend: row.get(6)?,
                        pid: row.get(7)?,
                        submitted_at: row.get(8)?,
                        finished_at: row.get(9)?,
                        exit_code: row.get(10)?,
                    })
                })
                .ok();

            match job {
                Some(j) => {
                    println!("Job: {}", j.id);
                    println!("  Script:     {}", j.script);
                    println!("  Status:     {}", j.status);
                    println!("  Framework:  {}", j.framework);
                    println!("  Backend:    {}", j.backend);
                    println!("  GPUs/node:  {}", j.gpus_per_node);
                    println!("  Nodes:      {}", j.nodes);
                    if let Some(pid) = j.pid {
                        println!("  PID:        {pid}");
                    }
                    println!("  Submitted:  {}", j.submitted_at);
                    if let Some(fin) = &j.finished_at {
                        println!("  Finished:   {fin}");
                    }
                    if let Some(exit) = j.exit_code {
                        println!("  Exit code:  {exit}");
                    }

                    // Show last 10 lines of log
                    let log_path = jobs_log_dir(&j.id).join("output.log");
                    if log_path.exists() {
                        println!();
                        println!("  Last output:");
                        if let Ok(content) = std::fs::read_to_string(&log_path) {
                            let lines: Vec<&str> = content.lines().collect();
                            let start = lines.len().saturating_sub(10);
                            for line in &lines[start..] {
                                println!("    {line}");
                            }
                        }
                    }
                }
                None => {
                    println!("Job not found: {id}");
                }
            }
        }

        JobCommands::Cancel { id } => {
            let conn = open_jobs_db()?;
            let pid: Option<u32> = conn
                .query_row(
                    "SELECT pid FROM jobs WHERE id = ?1 AND status = 'running'",
                    [&id],
                    |row| row.get(0),
                )
                .ok();

            match pid {
                Some(pid) if pid > 0 => {
                    println!("Cancelling job {id} (PID: {pid})...");

                    // Send SIGTERM (or taskkill on Windows)
                    #[cfg(unix)]
                    {
                        unsafe {
                            libc::kill(pid as i32, libc::SIGTERM);
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        let _ = std::process::Command::new("taskkill")
                            .args(["/PID", &pid.to_string(), "/F"])
                            .output();
                    }

                    conn.execute(
                        "UPDATE jobs SET status = 'cancelled', finished_at = ?1 WHERE id = ?2",
                        (Utc::now().to_rfc3339(), &id),
                    )?;

                    println!("Job cancelled.");
                }
                _ => {
                    println!("No running job found with ID: {id}");
                }
            }
        }
    }
    Ok(())
}
