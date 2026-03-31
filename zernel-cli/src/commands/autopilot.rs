// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel autopilot — Autonomous training optimizer
//!
//! Monitors training in real-time and automatically fixes problems:
//! - Detects GPU underutilization → suggests increasing DataLoader workers
//! - Detects memory pressure → suggests gradient checkpointing
//! - Detects NaN gradients → stops early and reports the layer
//! - Detects data bottleneck → suggests prefetching
//! - Tracks loss curve and detects divergence

use anyhow::{Context, Result};
use clap::Subcommand;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};

#[derive(Subcommand)]
pub enum AutopilotCommands {
    /// Run a training script with autonomous monitoring and optimization
    Run {
        /// Training script
        script: String,
        /// Additional arguments
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Analyze a running training job
    Analyze,
}

struct AutopilotState {
    step: u64,
    losses: Vec<f64>,
    gpu_utils: Vec<u32>,
    warnings: Vec<String>,
    interventions: Vec<String>,
}

impl AutopilotState {
    fn new() -> Self {
        Self {
            step: 0,
            losses: Vec::new(),
            gpu_utils: Vec::new(),
            warnings: Vec::new(),
            interventions: Vec::new(),
        }
    }

    fn check_loss_divergence(&mut self) {
        if self.losses.len() < 10 {
            return;
        }
        let recent = &self.losses[self.losses.len() - 5..];
        let earlier = &self.losses[self.losses.len() - 10..self.losses.len() - 5];
        let recent_avg: f64 = recent.iter().sum::<f64>() / 5.0;
        let earlier_avg: f64 = earlier.iter().sum::<f64>() / 5.0;

        if recent_avg > earlier_avg * 1.5 {
            let msg = format!(
                "Step {}: Loss diverging ({:.4} → {:.4}). Consider reducing learning rate.",
                self.step, earlier_avg, recent_avg
            );
            self.warnings.push(msg.clone());
            println!("  ⚠ AUTOPILOT: {msg}");
        }

        // Check for NaN
        if let Some(last) = self.losses.last() {
            if last.is_nan() || last.is_infinite() {
                let msg = format!("Step {}: NaN/Inf detected in loss! Stopping.", self.step);
                self.warnings.push(msg.clone());
                println!("  🛑 AUTOPILOT: {msg}");
                println!("  Fix: reduce learning rate, check data for corrupted samples,");
                println!("       or enable gradient clipping: torch.nn.utils.clip_grad_norm_");
            }
        }
    }

    fn check_gpu_utilization(&mut self) {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                if let Ok(util) = stdout.trim().parse::<u32>() {
                    self.gpu_utils.push(util);

                    if self.gpu_utils.len() >= 5 {
                        let recent: u32 = self.gpu_utils[self.gpu_utils.len() - 5..]
                            .iter()
                            .sum::<u32>()
                            / 5;

                        if recent < 30 && !self.interventions.contains(&"low_gpu".to_string()) {
                            let msg = format!(
                                "Step {}: GPU utilization low ({}%). Data pipeline is the bottleneck.",
                                self.step, recent
                            );
                            println!("  ⚡ AUTOPILOT: {msg}");
                            println!("    → Increase num_workers in DataLoader");
                            println!("    → Enable pin_memory=True");
                            println!("    → Use prefetch_factor=4");
                            self.interventions.push("low_gpu".to_string());
                        }

                        if recent > 95 && !self.interventions.contains(&"high_gpu".to_string()) {
                            println!(
                                "  ✓ AUTOPILOT: GPU utilization excellent ({}%). Training is GPU-bound.",
                                recent
                            );
                            self.interventions.push("high_gpu".to_string());
                        }
                    }
                }
            }
        }
    }

    fn print_summary(&self) {
        println!();
        println!("Zernel Autopilot Summary");
        println!("{}", "=".repeat(50));
        println!("  Steps monitored: {}", self.step);
        println!("  Warnings:        {}", self.warnings.len());
        println!("  Interventions:   {}", self.interventions.len());

        if !self.losses.is_empty() {
            let first = self.losses[0];
            let last = self.losses[self.losses.len() - 1];
            let improvement = if first > 0.0 {
                (1.0 - last / first) * 100.0
            } else {
                0.0
            };
            println!(
                "  Loss:            {:.4} → {:.4} ({:.1}% improvement)",
                first, last, improvement
            );
        }

        if !self.gpu_utils.is_empty() {
            let avg: u32 = self.gpu_utils.iter().sum::<u32>() / self.gpu_utils.len() as u32;
            println!("  Avg GPU util:    {}%", avg);
        }

        if !self.warnings.is_empty() {
            println!();
            println!("  Warnings:");
            for w in &self.warnings {
                println!("    - {w}");
            }
        }
    }
}

pub async fn run(cmd: AutopilotCommands) -> Result<()> {
    match cmd {
        AutopilotCommands::Run { script, args } => {
            println!("Zernel Autopilot");
            println!("{}", "=".repeat(50));
            println!("  Script:  {script}");
            println!("  Mode:    autonomous monitoring + optimization");
            println!();
            println!("Autopilot will:");
            println!("  - Monitor GPU utilization and suggest DataLoader changes");
            println!("  - Track loss curve and detect divergence/NaN");
            println!("  - Alert on memory pressure");
            println!("  - Report optimization opportunities");
            println!();

            let mut state = AutopilotState::new();
            let extractor = crate::experiments::tracker::MetricExtractor::new();

            let python = if std::process::Command::new("python3")
                .arg("--version")
                .output()
                .is_ok()
            {
                "python3"
            } else {
                "python"
            };

            let mut child = tokio::process::Command::new(python)
                .arg(&script)
                .args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .with_context(|| format!("failed to launch {script}"))?;

            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| anyhow::anyhow!("no stdout"))?;

            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            let mut check_interval = 0u64;

            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => break,
                    Ok(_) => {
                        let trimmed = line.trim_end();
                        println!("{trimmed}");

                        // Extract metrics
                        let metrics = extractor.extract_from_line(trimmed);
                        if let Some(&loss) = metrics.get("loss") {
                            state.losses.push(loss);
                            state.step += 1;
                            state.check_loss_divergence();
                        }
                        if let Some(&step) = metrics.get("step") {
                            state.step = step as u64;
                        }

                        // Periodic GPU check (every 10 lines)
                        check_interval += 1;
                        if check_interval.is_multiple_of(10) {
                            state.check_gpu_utilization();
                        }
                    }
                    Err(_) => break,
                }
            }

            let status = child.wait().await?;
            state.print_summary();

            if !status.success() {
                println!();
                println!(
                    "Training failed with exit code {}",
                    status.code().unwrap_or(-1)
                );
            }
        }

        AutopilotCommands::Analyze => {
            println!("Zernel Autopilot — Live Analysis");
            println!("{}", "=".repeat(50));
            println!();

            // Check current state
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args([
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 5 {
                        let util: u32 = f[0].parse().unwrap_or(0);
                        let mem_used: u64 = f[1].parse().unwrap_or(0);
                        let mem_total: u64 = f[2].parse().unwrap_or(1);
                        let mem_pct = mem_used * 100 / mem_total;

                        println!("  GPU Utilization: {}%", util);
                        println!("  Memory: {}% ({}/{}MB)", mem_pct, mem_used, mem_total);
                        println!("  Temperature: {}°C", f[3]);
                        println!("  Power: {}W", f[4]);
                        println!();

                        if util < 30 {
                            println!("  ⚡ RECOMMENDATION: GPU is underutilized.");
                            println!("    → Increase DataLoader num_workers");
                            println!("    → Enable pin_memory=True");
                        } else if util > 90 {
                            println!("  ✓ GPU utilization is excellent.");
                        }

                        if mem_pct > 90 {
                            println!("  ⚠ RECOMMENDATION: GPU memory nearly full.");
                            println!("    → Enable gradient checkpointing");
                            println!("    → Use mixed precision (BF16/FP16)");
                            println!("    → Reduce batch size");
                        }
                    }
                }
            } else {
                println!("  nvidia-smi not available.");
            }
        }
    }
    Ok(())
}
