// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use crate::phase_detector::PhaseDetectorConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level scheduler configuration, loaded from /etc/zernel/scheduler.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub phase_detection: PhaseDetectionConfig,
    #[serde(default)]
    pub numa: NumaConfig,
    #[serde(default)]
    pub multi_tenant: MultiTenantConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    /// How often (ms) to re-evaluate task phases.
    pub phase_eval_interval_ms: u64,
    /// How often (ms) to poll GPU utilization.
    pub gpu_poll_interval_ms: u64,
    /// Maximum number of tracked tasks.
    pub max_tracked_tasks: usize,
    /// Log level override (trace, debug, info, warn, error).
    pub log_level: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            phase_eval_interval_ms: 100,
            gpu_poll_interval_ms: 500,
            max_tracked_tasks: 65536,
            log_level: "info".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PhaseDetectionConfig {
    pub io_wait_threshold: f32,
    pub optimizer_burst_max_ns: u64,
    pub gpu_idle_threshold: u8,
    pub gpu_active_threshold: u8,
    /// Number of consecutive samples before committing a phase transition.
    pub phase_stability_count: u32,
    /// Enable NCCL collective detection (requires BPF probes).
    pub nccl_detection_enabled: bool,
}

impl Default for PhaseDetectionConfig {
    fn default() -> Self {
        Self {
            io_wait_threshold: 0.3,
            optimizer_burst_max_ns: 5_000_000,
            gpu_idle_threshold: 10,
            gpu_active_threshold: 80,
            phase_stability_count: 3,
            nccl_detection_enabled: false,
        }
    }
}

impl From<&PhaseDetectionConfig> for PhaseDetectorConfig {
    fn from(cfg: &PhaseDetectionConfig) -> Self {
        PhaseDetectorConfig {
            io_wait_threshold: cfg.io_wait_threshold,
            optimizer_burst_max_ns: cfg.optimizer_burst_max_ns,
            gpu_idle_threshold: cfg.gpu_idle_threshold,
            gpu_active_threshold: cfg.gpu_active_threshold,
            phase_stability_count: cfg.phase_stability_count,
            nccl_detection_enabled: cfg.nccl_detection_enabled,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NumaConfig {
    /// Enable NUMA-aware CPU selection for ML tasks.
    pub enabled: bool,
    /// Prefer CPUs on the same NUMA node as the task's GPU.
    pub gpu_affinity: bool,
    /// Prefer CPUs on the same NUMA node as the task's memory.
    pub memory_affinity: bool,
}

impl Default for NumaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            gpu_affinity: true,
            memory_affinity: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MultiTenantConfig {
    /// Enable multi-tenant scheduling (GPU-proportional CPU allocation).
    pub enabled: bool,
    /// Default priority class for new tasks.
    pub default_priority_class: String,
}

impl Default for MultiTenantConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_priority_class: "normal".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetryConfig {
    /// Expose scheduler metrics on this port (0 = disabled).
    pub metrics_port: u16,
    /// Push interval (ms) for telemetry export.
    pub push_interval_ms: u64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            metrics_port: 9093,
            push_interval_ms: 1000,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            phase_detection: PhaseDetectionConfig::default(),
            numa: NumaConfig::default(),
            multi_tenant: MultiTenantConfig::default(),
            telemetry: TelemetryConfig::default(),
        }
    }
}

impl SchedulerConfig {
    /// Load config from a TOML file, falling back to defaults for missing fields.
    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let config: Self = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }

    /// Generate a default config file.
    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_serializes() {
        let config = SchedulerConfig::default();
        let toml = config.to_toml().unwrap();
        assert!(toml.contains("phase_eval_interval_ms"));
        assert!(toml.contains("gpu_affinity"));
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        let config = SchedulerConfig::load(Path::new("/nonexistent/path.toml")).unwrap();
        assert_eq!(config.general.phase_eval_interval_ms, 100);
    }

    #[test]
    fn partial_toml_fills_defaults() {
        let toml = r#"
[general]
phase_eval_interval_ms = 200
"#;
        let config: SchedulerConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.general.phase_eval_interval_ms, 200);
        assert_eq!(config.general.gpu_poll_interval_ms, 500); // default
        assert!(config.numa.enabled); // default
    }
}
