// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Priority class for multi-tenant scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PriorityClass {
    /// Training jobs — highest resource allocation.
    Training,
    /// Inference serving — latency-sensitive but lower throughput needs.
    Inference,
    /// Interactive notebooks — responsive but best-effort.
    Interactive,
    /// Background — batch jobs, preprocessing, lowest priority.
    Background,
}

impl PriorityClass {
    /// Base priority modifier for this class.
    pub fn base_priority(&self) -> i32 {
        match self {
            Self::Training => 5,
            Self::Inference => 3,
            Self::Interactive => 1,
            Self::Background => -5,
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "training" => Self::Training,
            "inference" => Self::Inference,
            "interactive" => Self::Interactive,
            "background" => Self::Background,
            _ => Self::Training,
        }
    }
}

/// A tenant represents a user or job group sharing GPU server resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    pub id: String,
    pub gpu_count: u32,
    pub priority_class: PriorityClass,
    /// Explicit CPU weight override (None = auto from GPU count).
    pub cpu_weight: Option<f32>,
}

/// Multi-tenant scheduler that enforces GPU-proportional CPU allocation.
///
/// If Job A has 2 GPUs and Job B has 6 GPUs, Job B gets 3x the CPU scheduling
/// weight for data loading. This prevents one tenant's data loading from
/// starving another's NCCL collectives.
pub struct TenantScheduler {
    tenants: HashMap<String, Tenant>,
    /// Maps pid -> tenant_id for quick lookup.
    pid_tenant_map: HashMap<u32, String>,
    total_gpus: u32,
}

impl TenantScheduler {
    pub fn new() -> Self {
        Self {
            tenants: HashMap::new(),
            pid_tenant_map: HashMap::new(),
            total_gpus: 0,
        }
    }

    /// Register a new tenant.
    pub fn register_tenant(&mut self, tenant: Tenant) {
        self.total_gpus += tenant.gpu_count;
        debug!(
            tenant_id = tenant.id,
            gpus = tenant.gpu_count,
            class = ?tenant.priority_class,
            "registered tenant"
        );
        self.tenants.insert(tenant.id.clone(), tenant);
    }

    /// Remove a tenant.
    pub fn unregister_tenant(&mut self, tenant_id: &str) {
        if let Some(tenant) = self.tenants.remove(tenant_id) {
            self.total_gpus = self.total_gpus.saturating_sub(tenant.gpu_count);
            self.pid_tenant_map.retain(|_, tid| tid != tenant_id);
        }
    }

    /// Associate a process with a tenant.
    pub fn assign_pid(&mut self, pid: u32, tenant_id: &str) {
        self.pid_tenant_map.insert(pid, tenant_id.to_string());
    }

    /// Get the CPU weight for a given pid.
    /// Weight is proportional to the tenant's GPU share.
    pub fn cpu_weight_for_pid(&self, pid: u32) -> f32 {
        let tenant_id = match self.pid_tenant_map.get(&pid) {
            Some(tid) => tid,
            None => return 1.0,
        };
        let tenant = match self.tenants.get(tenant_id) {
            Some(t) => t,
            None => return 1.0,
        };

        // Explicit override
        if let Some(w) = tenant.cpu_weight {
            return w;
        }

        // GPU-proportional weight
        if self.total_gpus == 0 {
            return 1.0;
        }

        (tenant.gpu_count as f32) / (self.total_gpus as f32)
            * self.tenants.len() as f32 // normalize so average weight = 1.0
    }

    /// Compute effective priority for a pid, combining phase priority with tenant weight.
    pub fn effective_priority(&self, pid: u32, phase_priority: i32) -> i32 {
        let weight = self.cpu_weight_for_pid(pid);
        let class_priority = self
            .pid_tenant_map
            .get(&pid)
            .and_then(|tid| self.tenants.get(tid))
            .map(|t| t.priority_class.base_priority())
            .unwrap_or(0);

        let weighted = (phase_priority as f32 * weight) as i32;
        weighted + class_priority
    }

    pub fn tenant_count(&self) -> usize {
        self.tenants.len()
    }

    pub fn get_tenant_for_pid(&self, pid: u32) -> Option<&Tenant> {
        self.pid_tenant_map
            .get(&pid)
            .and_then(|tid| self.tenants.get(tid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_proportional_weight() {
        let mut ts = TenantScheduler::new();
        ts.register_tenant(Tenant {
            id: "job-a".into(),
            gpu_count: 2,
            priority_class: PriorityClass::Training,
            cpu_weight: None,
        });
        ts.register_tenant(Tenant {
            id: "job-b".into(),
            gpu_count: 6,
            priority_class: PriorityClass::Training,
            cpu_weight: None,
        });

        ts.assign_pid(100, "job-a");
        ts.assign_pid(200, "job-b");

        let w_a = ts.cpu_weight_for_pid(100);
        let w_b = ts.cpu_weight_for_pid(200);
        // Job B should get 3x the weight of Job A
        assert!((w_b / w_a - 3.0).abs() < 0.01);
    }

    #[test]
    fn priority_class_affects_priority() {
        let mut ts = TenantScheduler::new();
        ts.register_tenant(Tenant {
            id: "train".into(),
            gpu_count: 4,
            priority_class: PriorityClass::Training,
            cpu_weight: Some(1.0),
        });
        ts.register_tenant(Tenant {
            id: "bg".into(),
            gpu_count: 4,
            priority_class: PriorityClass::Background,
            cpu_weight: Some(1.0),
        });

        ts.assign_pid(100, "train");
        ts.assign_pid(200, "bg");

        let p_train = ts.effective_priority(100, 10);
        let p_bg = ts.effective_priority(200, 10);
        assert!(p_train > p_bg);
    }

    #[test]
    fn unregistered_pid_gets_default_weight() {
        let ts = TenantScheduler::new();
        assert_eq!(ts.cpu_weight_for_pid(999), 1.0);
    }

    #[test]
    fn unregister_tenant_cleans_up() {
        let mut ts = TenantScheduler::new();
        ts.register_tenant(Tenant {
            id: "job-a".into(),
            gpu_count: 4,
            priority_class: PriorityClass::Training,
            cpu_weight: None,
        });
        ts.assign_pid(100, "job-a");
        assert_eq!(ts.tenant_count(), 1);

        ts.unregister_tenant("job-a");
        assert_eq!(ts.tenant_count(), 0);
        assert_eq!(ts.cpu_weight_for_pid(100), 1.0); // cleaned up
    }
}
