// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::{Deserialize, Serialize};
use tracing::warn;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub metric: String,
    pub threshold: f64,
    pub comparison: Comparison,
    pub action: AlertAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Comparison {
    GreaterThan,
    LessThan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertAction {
    Log,
    Webhook { url: String },
}

pub struct AlertEngine {
    rules: Vec<AlertRule>,
}

impl AlertEngine {
    pub fn new(rules: Vec<AlertRule>) -> Self {
        Self { rules }
    }

    pub fn evaluate(&self, metric_name: &str, value: f64) {
        for rule in &self.rules {
            if rule.metric != metric_name {
                continue;
            }
            let triggered = match rule.comparison {
                Comparison::GreaterThan => value > rule.threshold,
                Comparison::LessThan => value < rule.threshold,
            };
            if triggered {
                warn!(
                    alert = rule.name,
                    metric = metric_name,
                    value,
                    threshold = rule.threshold,
                    "alert triggered"
                );
            }
        }
    }
}
