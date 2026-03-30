// Copyright (C) 2026 Dyber, Inc. — Proprietary

use crate::experiments::store::ExperimentStore;
use crate::experiments::tracker;
use anyhow::Result;
use std::path::PathBuf;

/// Show training logs for an experiment.
pub async fn run(id: Option<String>, follow: bool, grep: Option<String>) -> Result<()> {
    let db_path = tracker::experiments_db_path();
    let store = ExperimentStore::open(&db_path)?;

    // Resolve experiment ID
    let exp_id = match id {
        Some(id) => id,
        None => {
            // Get the latest experiment
            let exps = store.list(1)?;
            match exps.first() {
                Some(exp) => exp.id.clone(),
                None => {
                    println!("No experiments yet. Run `zernel run <script>` first.");
                    return Ok(());
                }
            }
        }
    };

    // Verify experiment exists
    let exp = store
        .get(&exp_id)?
        .ok_or_else(|| anyhow::anyhow!("experiment not found: {exp_id}"))?;

    let log_path = experiment_log_path(&exp_id);

    println!("Log for experiment: {} ({})", exp.id, exp.name);
    println!("  Status: {}", exp.status);
    if let Some(d) = exp.duration_secs {
        println!("  Duration: {d:.1}s");
    }
    println!("  Log file: {}", log_path.display());
    println!();

    if !log_path.exists() {
        println!("(no log file found — logs are saved for experiments run with zernel >= v0.1.0)");
        return Ok(());
    }

    let content = std::fs::read_to_string(&log_path)?;

    if let Some(ref pattern) = grep {
        // Filtered output
        let mut found = 0;
        for (i, line) in content.lines().enumerate() {
            if line.contains(pattern.as_str()) {
                println!("{:>6}: {line}", i + 1);
                found += 1;
            }
        }
        if found == 0 {
            println!("No lines matching '{pattern}'");
        } else {
            println!("\n{found} matching line(s)");
        }
    } else if follow && exp.status.to_string() == "Running" {
        // Follow mode for active experiments
        println!("--- following (Ctrl+C to stop) ---");
        println!();

        // Print existing content
        print!("{content}");

        // Tail the file
        let mut last_len = content.len();
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            if let Ok(new_content) = std::fs::read_to_string(&log_path) {
                if new_content.len() > last_len {
                    print!("{}", &new_content[last_len..]);
                    last_len = new_content.len();
                }
            }
        }
    } else {
        // Full output
        print!("{content}");
    }

    Ok(())
}

/// Get the log file path for an experiment.
pub fn experiment_log_path(exp_id: &str) -> PathBuf {
    tracker::zernel_dir()
        .join("experiments")
        .join(exp_id)
        .join("output.log")
}
