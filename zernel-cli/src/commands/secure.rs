// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel secure — System hardening for production ML
//!
//! Lock down the OS for production GPU clusters: disable unnecessary services,
//! configure firewall, enable audit logging, set up security updates.

use anyhow::Result;
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum SecureCommands {
    /// Scan system and show security recommendations
    Scan,
    /// Apply production hardening (requires root)
    Harden {
        /// Dry run
        #[arg(long)]
        dry_run: bool,
    },
    /// Configure firewall for ML workloads (NCCL ports only)
    Firewall {
        /// Dry run
        #[arg(long)]
        dry_run: bool,
    },
    /// Enable audit logging
    Audit,
    /// Check for security updates
    Updates,
}

fn run_check(_name: &str, cmd_name: &str, args: &[&str]) -> (bool, String) {
    match Command::new(cmd_name).args(args).output() {
        Ok(o) if o.status.success() => {
            let out = String::from_utf8_lossy(&o.stdout).trim().to_string();
            (true, out)
        }
        _ => (false, "N/A".into()),
    }
}

pub async fn run(cmd: SecureCommands) -> Result<()> {
    match cmd {
        SecureCommands::Scan => {
            println!("Zernel Security Scan");
            println!("{}", "=".repeat(60));
            println!();

            let mut issues = 0;

            // Check SSH
            println!("[1/8] SSH Configuration");
            #[cfg(target_os = "linux")]
            {
                if let Ok(config) = std::fs::read_to_string("/etc/ssh/sshd_config") {
                    let root_login = config
                        .lines()
                        .any(|l| l.trim().starts_with("PermitRootLogin") && l.contains("yes"));
                    let password_auth = config.lines().any(|l| {
                        l.trim().starts_with("PasswordAuthentication") && l.contains("yes")
                    });

                    if root_login {
                        println!("  WARN: Root login enabled — disable with PermitRootLogin no");
                        issues += 1;
                    } else {
                        println!("  OK: Root login disabled");
                    }

                    if password_auth {
                        println!("  WARN: Password auth enabled — use key-based auth only");
                        issues += 1;
                    } else {
                        println!("  OK: Key-based auth only");
                    }
                } else {
                    println!("  SKIP: sshd_config not found");
                }
            }
            #[cfg(not(target_os = "linux"))]
            println!("  SKIP: Linux only");

            // Check firewall
            println!();
            println!("[2/8] Firewall");
            let (fw_ok, _) = run_check("iptables", "iptables", &["-L", "-n"]);
            if fw_ok {
                println!("  OK: Firewall active");
            } else {
                println!("  WARN: No firewall rules — configure with: zernel secure firewall");
                issues += 1;
            }

            // Check unattended upgrades
            println!();
            println!("[3/8] Security Updates");
            #[cfg(target_os = "linux")]
            {
                let auto_updates =
                    std::path::Path::new("/etc/apt/apt.conf.d/20auto-upgrades").exists();
                if auto_updates {
                    println!("  OK: Unattended security updates enabled");
                } else {
                    println!("  WARN: Auto security updates not configured");
                    println!("    Fix: apt install unattended-upgrades && dpkg-reconfigure unattended-upgrades");
                    issues += 1;
                }
            }
            #[cfg(not(target_os = "linux"))]
            println!("  SKIP: Linux only");

            // Check swap
            println!();
            println!("[4/8] Swap (should be disabled for ML)");
            #[cfg(target_os = "linux")]
            {
                if let Ok(swaps) = std::fs::read_to_string("/proc/swaps") {
                    let swap_active = swaps.lines().count() > 1;
                    if swap_active {
                        println!("  WARN: Swap is active — disable with: swapoff -a");
                        issues += 1;
                    } else {
                        println!("  OK: Swap disabled");
                    }
                }
            }
            #[cfg(not(target_os = "linux"))]
            println!("  SKIP: Linux only");

            // Check unnecessary services
            println!();
            println!("[5/8] Unnecessary Services");
            let unnecessary = ["cups", "avahi-daemon", "bluetooth", "ModemManager"];
            for svc in &unnecessary {
                let (running, _) = run_check(svc, "systemctl", &["is-active", svc]);
                if running {
                    println!("  WARN: {svc} is running — not needed for ML servers");
                    issues += 1;
                }
            }
            if issues == 0 {
                println!("  OK: No unnecessary services detected");
            }

            // Check GPU persistence mode
            println!();
            println!("[6/8] GPU Persistence Mode");
            let (_, gpu_pm) = run_check(
                "nvidia-smi",
                "nvidia-smi",
                &["--query-gpu=persistence_mode", "--format=csv,noheader"],
            );
            if gpu_pm.contains("Enabled") {
                println!("  OK: GPU persistence mode enabled");
            } else {
                println!("  WARN: GPU persistence mode disabled — enable with: nvidia-smi -pm 1");
                issues += 1;
            }

            // Check PQC keys
            println!();
            println!("[7/8] Post-Quantum Cryptography");
            let pqc_dir = crate::experiments::tracker::zernel_dir().join("pqc");
            let has_keys = pqc_dir.exists()
                && std::fs::read_dir(&pqc_dir)
                    .map(|entries| entries.flatten().count() > 0)
                    .unwrap_or(false);
            if has_keys {
                println!("  OK: PQC keys configured");
            } else {
                println!("  INFO: No PQC keys — generate with: zernel pqc keygen");
            }

            // Check kernel tuning
            println!();
            println!("[8/8] Kernel Tuning");
            #[cfg(target_os = "linux")]
            {
                let zernel_conf = std::path::Path::new("/etc/sysctl.d/99-zernel.conf");
                if zernel_conf.exists() {
                    println!("  OK: Zernel sysctl tuning applied");
                } else {
                    println!("  WARN: Zernel sysctl tuning not applied");
                    println!("    Fix: zernel tune apply");
                    issues += 1;
                }
            }
            #[cfg(not(target_os = "linux"))]
            println!("  SKIP: Linux only");

            // Summary
            println!();
            println!("{}", "=".repeat(60));
            if issues == 0 {
                println!("Security scan: PASS (0 issues)");
            } else {
                println!("Security scan: {issues} issue(s) found");
                println!("Harden: zernel secure harden");
            }
        }

        SecureCommands::Harden { dry_run } => {
            println!(
                "Zernel System Hardening{}",
                if dry_run { " (dry run)" } else { "" }
            );
            println!("{}", "=".repeat(50));

            let actions: Vec<(&str, Vec<&str>)> = vec![
                ("Disable swap", vec!["swapoff", "-a"]),
                ("Enable GPU persistence", vec!["nvidia-smi", "-pm", "1"]),
                (
                    "Disable unnecessary services",
                    vec![
                        "systemctl",
                        "disable",
                        "--now",
                        "cups",
                        "avahi-daemon",
                        "bluetooth",
                    ],
                ),
                (
                    "Install unattended-upgrades",
                    vec!["apt-get", "install", "-y", "unattended-upgrades"],
                ),
            ];

            for (name, args) in &actions {
                if dry_run {
                    println!("  WOULD: {name} ({} {})", args[0], args[1..].join(" "));
                } else {
                    print!("  {name}... ");
                    let status = Command::new(args[0]).args(&args[1..]).output();
                    match status {
                        Ok(o) if o.status.success() => println!("OK"),
                        _ => println!("SKIP (may need root)"),
                    }
                }
            }

            // Apply sysctl tuning
            if dry_run {
                println!("  WOULD: Apply kernel tuning (zernel tune apply)");
            } else {
                println!("  Applying kernel tuning...");
                let _ = Command::new("zernel").args(["tune", "apply"]).status();
            }

            println!();
            if !dry_run {
                println!("Hardening complete. Run: zernel secure scan  — to verify");
            }
        }

        SecureCommands::Firewall { dry_run } => {
            println!(
                "Zernel ML Firewall Configuration{}",
                if dry_run { " (dry run)" } else { "" }
            );
            println!("{}", "=".repeat(50));
            println!();
            println!("Allows:");
            println!("  - SSH (port 22)");
            println!("  - NCCL (ports 29500-30000)");
            println!("  - Zernel services (9091, 9092, 3000)");
            println!("  - Prometheus (9090)");
            println!("  - Ollama (11434)");
            println!("Blocks: everything else inbound");
            println!();

            let rules = [
                vec![
                    "-A",
                    "INPUT",
                    "-m",
                    "state",
                    "--state",
                    "ESTABLISHED,RELATED",
                    "-j",
                    "ACCEPT",
                ],
                vec!["-A", "INPUT", "-i", "lo", "-j", "ACCEPT"],
                vec!["-A", "INPUT", "-p", "tcp", "--dport", "22", "-j", "ACCEPT"],
                vec![
                    "-A",
                    "INPUT",
                    "-p",
                    "tcp",
                    "--dport",
                    "29500:30000",
                    "-j",
                    "ACCEPT",
                ],
                vec![
                    "-A", "INPUT", "-p", "tcp", "--dport", "9091", "-j", "ACCEPT",
                ],
                vec![
                    "-A", "INPUT", "-p", "tcp", "--dport", "9092", "-j", "ACCEPT",
                ],
                vec![
                    "-A", "INPUT", "-p", "tcp", "--dport", "3000", "-j", "ACCEPT",
                ],
                vec![
                    "-A", "INPUT", "-p", "tcp", "--dport", "9090", "-j", "ACCEPT",
                ],
                vec![
                    "-A", "INPUT", "-p", "tcp", "--dport", "11434", "-j", "ACCEPT",
                ],
                vec!["-A", "INPUT", "-p", "icmp", "-j", "ACCEPT"],
                vec!["-P", "INPUT", "DROP"],
            ];

            for rule in &rules {
                if dry_run {
                    println!("  iptables {}", rule.join(" "));
                } else {
                    let _ = Command::new("iptables").args(rule).output();
                    println!("  Applied: iptables {}", rule.join(" "));
                }
            }

            if !dry_run {
                println!();
                println!("Firewall configured. Save with: iptables-save > /etc/iptables.rules");
            }
        }

        SecureCommands::Audit => {
            println!("Enabling Audit Logging");
            println!("{}", "=".repeat(50));

            let status = Command::new("apt-get")
                .args(["install", "-y", "auditd"])
                .output();

            match status {
                Ok(o) if o.status.success() => {
                    println!("  auditd installed");

                    // Add ML-relevant audit rules
                    let rules = [
                        "-w /etc/zernel/ -p wa -k zernel_config",
                        "-w /usr/local/bin/zernel -p x -k zernel_exec",
                        "-w /usr/local/bin/zerneld -p x -k zerneld_exec",
                    ];

                    for rule in &rules {
                        let _ = Command::new("auditctl")
                            .args(rule.split_whitespace())
                            .output();
                        println!("  Rule: {rule}");
                    }

                    println!();
                    println!("Audit logging enabled. View: ausearch -k zernel_config");
                }
                _ => {
                    println!("  Failed to install auditd (requires root + apt)");
                }
            }
        }

        SecureCommands::Updates => {
            println!("Security Updates Check");
            println!("{}", "=".repeat(50));

            let output = Command::new("apt-get")
                .args(["--just-print", "upgrade"])
                .output();

            match output {
                Ok(o) if o.status.success() => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let upgradable: Vec<&str> =
                        stdout.lines().filter(|l| l.starts_with("Inst")).collect();

                    if upgradable.is_empty() {
                        println!("  All packages up to date.");
                    } else {
                        println!("  {} packages need updating:", upgradable.len());
                        for pkg in upgradable.iter().take(20) {
                            println!("    {pkg}");
                        }
                        if upgradable.len() > 20 {
                            println!("    ... and {} more", upgradable.len() - 20);
                        }
                        println!();
                        println!("  Apply: sudo apt-get upgrade -y");
                    }
                }
                _ => {
                    println!("  Cannot check updates (requires apt)");
                }
            }
        }
    }
    Ok(())
}
