// Copyright (C) 2026 Dyber, Inc. — Proprietary

use std::collections::HashMap;

/// Extracts metrics from training script stdout.
///
/// Recognizes common patterns:
/// - `loss: 1.234` or `loss=1.234`
/// - `accuracy: 0.95` or `acc=0.95`
/// - tqdm-style progress bars with metric suffixes
/// - HuggingFace Trainer log format
pub struct MetricExtractor {
    patterns: Vec<MetricPattern>,
}

struct MetricPattern {
    name: String,
    regex: String, // Will use regex crate when implemented
}

impl MetricExtractor {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                MetricPattern {
                    name: "loss".into(),
                    regex: r"loss[=:\s]+([0-9]+\.?[0-9]*)".into(),
                },
                MetricPattern {
                    name: "accuracy".into(),
                    regex: r"(?:accuracy|acc)[=:\s]+([0-9]+\.?[0-9]*)".into(),
                },
                MetricPattern {
                    name: "grad_norm".into(),
                    regex: r"grad_norm[=:\s]+([0-9]+\.?[0-9]*)".into(),
                },
                MetricPattern {
                    name: "learning_rate".into(),
                    regex: r"(?:learning_rate|lr)[=:\s]+([0-9]+\.?[0-9eE\-]*)".into(),
                },
            ],
        }
    }

    /// Parse a line of stdout and extract any recognized metrics.
    pub fn extract_from_line(&self, _line: &str) -> HashMap<String, f64> {
        // TODO: Implement regex matching against self.patterns
        HashMap::new()
    }
}

/// Generates a unique experiment ID.
pub fn generate_experiment_id() -> String {
    let now = chrono::Utc::now();
    format!("exp-{}", now.format("%Y%m%d-%H%M%S"))
}
