// Copyright (C) 2026 Dyber, Inc. — Proprietary

use crate::experiments::store::{Experiment, ExperimentStatus, ExperimentStore};
use crate::experiments::tracker::{self, MetricExtractor};
use anyhow::{Context, Result};
use chrono::Utc;
use std::collections::HashMap;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Run a training script with automatic telemetry and experiment tracking.
pub async fn run(script: &str, args: &[String]) -> Result<()> {
    let db_path = tracker::experiments_db_path();
    let store = ExperimentStore::open(&db_path)?;
    let extractor = MetricExtractor::new();

    let exp_id = tracker::generate_experiment_id();
    let exp_name = std::path::Path::new(script)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unnamed".into());

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

    let experiment = Experiment {
        id: exp_id.clone(),
        name: exp_name.clone(),
        status: ExperimentStatus::Running,
        hyperparams: HashMap::new(),
        metrics: HashMap::new(),
        created_at: Utc::now(),
        finished_at: None,
        git_commit,
        script: Some(script.to_string()),
        duration_secs: None,
    };

    store.insert(&experiment)?;

    println!("Zernel Run");
    println!("  Experiment: {exp_id}");
    println!("  Script:     {script}");
    println!("  Tracking:   enabled");
    println!();

    let python = detect_python();
    let start = std::time::Instant::now();

    let mut child = tokio::process::Command::new(&python)
        .arg(script)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to launch: {python} {script}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture child stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture child stderr"))?;

    let exp_id_clone = exp_id.clone();
    let db_path_clone = db_path.clone();

    // Process stdout in a spawned task
    let stdout_handle = tokio::spawn(async move {
        let mut latest_metrics: HashMap<String, f64> = HashMap::new();
        let mut lines_processed = 0u64;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();

        let store = ExperimentStore::open(&db_path_clone).ok();

        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {
                    let trimmed = line.trim_end();
                    println!("{trimmed}");

                    let extracted = extractor.extract_from_line(trimmed);
                    if !extracted.is_empty() {
                        for (k, v) in &extracted {
                            latest_metrics.insert(k.clone(), *v);
                        }
                        lines_processed += 1;

                        if lines_processed.is_multiple_of(10) {
                            if let Some(ref s) = store {
                                let _ = s.update_metrics(&exp_id_clone, &latest_metrics);
                            }
                        }
                    }
                }
                Err(_) => break,
            }
        }
        latest_metrics
    });

    // Forward stderr
    let stderr_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {
                    eprint!("{}", line);
                }
                Err(_) => break,
            }
        }
    });

    let (latest_metrics, _) = tokio::join!(stdout_handle, stderr_handle);
    let latest_metrics = latest_metrics.unwrap_or_default();

    let status = child.wait().await?;
    let duration = start.elapsed();

    let final_status = if status.success() {
        ExperimentStatus::Done
    } else {
        ExperimentStatus::Failed
    };

    store.update_metrics(&exp_id, &latest_metrics)?;
    store.finish(&exp_id, final_status.clone(), duration.as_secs_f64())?;

    println!();
    println!("---");
    println!("  Status:    {final_status}");
    println!("  Duration:  {:.1}s", duration.as_secs_f64());
    println!("  Experiment: {exp_id}");

    if !latest_metrics.is_empty() {
        println!("  Metrics:");
        let mut sorted: Vec<_> = latest_metrics.iter().collect();
        sorted.sort_by_key(|(k, _)| (*k).clone());
        for (k, v) in sorted {
            println!("    {k}: {v:.4}");
        }
    }

    println!();
    println!("View: zernel exp show {exp_id}");

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}

fn detect_python() -> String {
    for candidate in &["python3", "python"] {
        if std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok()
        {
            return candidate.to_string();
        }
    }
    "python3".to_string()
}
