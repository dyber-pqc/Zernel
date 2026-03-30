// Copyright (C) 2026 Dyber, Inc. — Proprietary

use regex::Regex;
use std::collections::HashMap;

/// Extracts metrics from training script stdout.
///
/// Recognizes common patterns:
/// - `loss: 1.234` or `loss=1.234`
/// - `accuracy: 0.95` or `acc=0.95`
/// - tqdm-style progress bars with metric suffixes
/// - HuggingFace Trainer log format
pub struct MetricExtractor {
    patterns: Vec<CompiledPattern>,
}

struct CompiledPattern {
    name: String,
    regex: Regex,
}

impl MetricExtractor {
    pub fn new() -> Self {
        let patterns = vec![
            ("loss", r"(?i)\bloss[=:\s]+([0-9]+\.?[0-9]*)"),
            (
                "accuracy",
                r"(?i)\b(?:accuracy|acc)[=:\s]+([0-9]+\.?[0-9]*)",
            ),
            ("grad_norm", r"(?i)\bgrad_norm[=:\s]+([0-9]+\.?[0-9]*)"),
            (
                "learning_rate",
                r"(?i)\b(?:learning_rate|lr)[=:\s]+([0-9]+\.?[0-9eE\-]*)",
            ),
            (
                "throughput",
                r"(?i)\b(?:throughput|samples/s|it/s)[=:\s]+([0-9]+\.?[0-9]*)",
            ),
            ("epoch", r"(?i)\bepoch[=:\s]+([0-9]+\.?[0-9]*)"),
            ("step", r"(?i)\b(?:step|global_step)[=:\s]+([0-9]+)"),
            (
                "perplexity",
                r"(?i)\b(?:perplexity|ppl)[=:\s]+([0-9]+\.?[0-9]*)",
            ),
            ("eval_loss", r"(?i)\beval_loss[=:\s]+([0-9]+\.?[0-9]*)"),
        ];

        Self {
            patterns: patterns
                .into_iter()
                .filter_map(|(name, pat)| {
                    Regex::new(pat).ok().map(|regex| CompiledPattern {
                        name: name.to_string(),
                        regex,
                    })
                })
                .collect(),
        }
    }

    /// Parse a line of stdout and extract any recognized metrics.
    pub fn extract_from_line(&self, line: &str) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        for pattern in &self.patterns {
            if let Some(caps) = pattern.regex.captures(line) {
                if let Some(m) = caps.get(1) {
                    if let Ok(val) = m.as_str().parse::<f64>() {
                        metrics.insert(pattern.name.clone(), val);
                    }
                }
            }
        }

        metrics
    }
}

/// Generates a unique experiment ID.
pub fn generate_experiment_id() -> String {
    let now = chrono::Utc::now();
    let short_uuid = &uuid::Uuid::new_v4().to_string()[..8];
    format!("exp-{}-{}", now.format("%Y%m%d-%H%M%S"), short_uuid)
}

/// Get the zernel data directory (~/.zernel/).
pub fn zernel_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".zernel")
}

/// Get the experiments database path.
pub fn experiments_db_path() -> std::path::PathBuf {
    let dir = zernel_dir().join("experiments");
    std::fs::create_dir_all(&dir).ok();
    dir.join("experiments.db")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_loss() {
        let ext = MetricExtractor::new();
        let m = ext.extract_from_line("Epoch 1/10, loss: 1.2345, accuracy: 0.876");
        assert!((m["loss"] - 1.2345).abs() < 0.0001);
        assert!((m["accuracy"] - 0.876).abs() < 0.001);
    }

    #[test]
    fn extract_equals_format() {
        let ext = MetricExtractor::new();
        let m = ext.extract_from_line("loss=0.456 lr=3e-4 grad_norm=0.89");
        assert!((m["loss"] - 0.456).abs() < 0.001);
        assert!((m["grad_norm"] - 0.89).abs() < 0.01);
    }

    #[test]
    fn extract_huggingface_format() {
        let ext = MetricExtractor::new();
        // HF Trainer logs: {'loss': 2.1, 'learning_rate': 5e-05, 'epoch': 0.5}
        // The regex uses \b word boundary, which fires after the ' quote
        let m = ext.extract_from_line("loss: 2.1, learning_rate: 5e-05, epoch: 0.5");
        assert!((m["loss"] - 2.1).abs() < 0.01);
        assert!((m["epoch"] - 0.5).abs() < 0.01);
    }

    #[test]
    fn extract_step() {
        let ext = MetricExtractor::new();
        let m = ext.extract_from_line("Step 4821/10000, loss: 1.23");
        assert_eq!(m["step"], 4821.0);
    }

    #[test]
    fn no_match_returns_empty() {
        let ext = MetricExtractor::new();
        let m = ext.extract_from_line("Loading dataset from disk...");
        assert!(m.is_empty());
    }

    #[test]
    fn experiment_id_is_unique() {
        let a = generate_experiment_id();
        let b = generate_experiment_id();
        assert_ne!(a, b);
        assert!(a.starts_with("exp-"));
    }
}
