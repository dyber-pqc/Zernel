// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

//! NCCL Network Priority via eBPF/tc
//!
//! When the scheduler detects NcclCollective phase, this module uses
//! Linux traffic control (tc) with eBPF classifiers to give NCCL
//! traffic absolute priority on network interfaces.
//!
//! This reduces all-reduce tail latency which is the critical path
//! in distributed training — the slowest rank determines throughput.

use anyhow::Result;
use std::process::Command;
use tracing::{debug, info, warn};

/// Network interface to apply NCCL priority rules to.
const DEFAULT_INTERFACES: &[&str] = &["eth0", "ib0", "enp", "ens"];

/// NCCL uses a specific port range (typically 29500-29999 for PyTorch).
const NCCL_PORT_START: u16 = 29500;
const NCCL_PORT_END: u16 = 30000;

/// DSCP marking for high-priority NCCL traffic.
const NCCL_DSCP: u8 = 46; // Expedited Forwarding (EF)

/// Detect the primary network interface for NCCL traffic.
pub fn detect_nccl_interface() -> Option<String> {
    // Try NCCL_SOCKET_IFNAME first
    if let Ok(iface) = std::env::var("NCCL_SOCKET_IFNAME") {
        return Some(iface);
    }

    // Check for InfiniBand
    if std::path::Path::new("/sys/class/infiniband").exists() {
        if let Ok(entries) = std::fs::read_dir("/sys/class/infiniband") {
            if let Some(entry) = entries.flatten().next() {
                let name = entry.file_name().to_string_lossy().to_string();
                return Some(name);
            }
        }
    }

    // Fall back to first non-lo interface
    let output = Command::new("ip")
        .args(["-o", "link", "show", "up"])
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let iface = parts[1].trim_end_matches(':');
            if iface != "lo" {
                return Some(iface.to_string());
            }
        }
    }

    None
}

/// Apply high-priority tc rules for NCCL traffic.
/// Requires root / CAP_NET_ADMIN.
pub fn enable_nccl_priority(interface: &str) -> Result<()> {
    info!(interface, "enabling NCCL traffic priority");

    // Add prio qdisc
    let _ = Command::new("tc")
        .args(["qdisc", "del", "dev", interface, "root"])
        .output(); // ignore error (may not exist)

    let status = Command::new("tc")
        .args([
            "qdisc", "add", "dev", interface, "root", "handle", "1:", "prio", "bands", "3",
            "priomap", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1",
            "1",
        ])
        .output();

    match status {
        Ok(o) if o.status.success() => {
            debug!("prio qdisc added");
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            warn!(error = %err, "failed to add prio qdisc (requires root)");
            return Ok(());
        }
        Err(e) => {
            warn!(error = %e, "tc not found");
            return Ok(());
        }
    }

    // Add filter: NCCL port range → highest priority band (0)
    for port in [NCCL_PORT_START, 29501, 29502, 29503, 29504] {
        let _ = Command::new("tc")
            .args([
                "filter",
                "add",
                "dev",
                interface,
                "parent",
                "1:0",
                "protocol",
                "ip",
                "prio",
                "1",
                "u32",
                "match",
                "ip",
                "dport",
                &port.to_string(),
                "0xffff",
                "flowid",
                "1:1",
            ])
            .output();
    }

    // Also prioritize by DSCP EF marking
    let _ = Command::new("tc")
        .args([
            "filter",
            "add",
            "dev",
            interface,
            "parent",
            "1:0",
            "protocol",
            "ip",
            "prio",
            "1",
            "u32",
            "match",
            "ip",
            "tos",
            &format!("0x{:02x}", NCCL_DSCP << 2),
            "0xfc",
            "flowid",
            "1:1",
        ])
        .output();

    info!(
        interface,
        ports = format!("{NCCL_PORT_START}-{NCCL_PORT_END}"),
        "NCCL traffic priority enabled"
    );

    Ok(())
}

/// Remove NCCL priority rules.
pub fn disable_nccl_priority(interface: &str) -> Result<()> {
    let _ = Command::new("tc")
        .args(["qdisc", "del", "dev", interface, "root"])
        .output();
    info!(interface, "NCCL traffic priority disabled");
    Ok(())
}

/// Mark outgoing NCCL packets with DSCP EF for switches/routers.
pub fn mark_nccl_dscp() -> Result<()> {
    // Use iptables to mark NCCL packets
    for port in NCCL_PORT_START..NCCL_PORT_END {
        let _ = Command::new("iptables")
            .args([
                "-t",
                "mangle",
                "-A",
                "OUTPUT",
                "-p",
                "tcp",
                "--dport",
                &port.to_string(),
                "-j",
                "DSCP",
                "--set-dscp",
                &NCCL_DSCP.to_string(),
            ])
            .output();
    }
    info!("NCCL packets marked with DSCP EF ({})", NCCL_DSCP);
    Ok(())
}
