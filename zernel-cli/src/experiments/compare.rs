// Copyright (C) 2026 Dyber, Inc. — Proprietary

use super::store::Experiment;
use std::collections::BTreeSet;

/// Compare two experiments and produce a human-readable diff.
pub fn compare(a: &Experiment, b: &Experiment) -> String {
    let mut out = String::new();

    out.push_str(&format!("Comparing: {} vs {}\n", a.id, b.id));
    out.push_str(&format!("          {} vs {}\n\n", a.name, b.name));

    // Hyperparameter diff
    let all_keys: BTreeSet<_> = a
        .hyperparams
        .keys()
        .chain(b.hyperparams.keys())
        .cloned()
        .collect();

    if !all_keys.is_empty() {
        out.push_str("Hyperparameters:\n");
        for key in &all_keys {
            let va = a.hyperparams.get(key).map(|v| v.to_string()).unwrap_or_default();
            let vb = b.hyperparams.get(key).map(|v| v.to_string()).unwrap_or_default();
            if va != vb {
                out.push_str(&format!("  {key}: {va} -> {vb}\n"));
            }
        }
        out.push('\n');
    }

    // Metrics diff
    let all_metrics: BTreeSet<_> = a
        .metrics
        .keys()
        .chain(b.metrics.keys())
        .cloned()
        .collect();

    if !all_metrics.is_empty() {
        out.push_str("Metrics:\n");
        for key in &all_metrics {
            let va = a.metrics.get(key).copied();
            let vb = b.metrics.get(key).copied();
            match (va, vb) {
                (Some(a_val), Some(b_val)) => {
                    let pct = if a_val != 0.0 {
                        ((b_val - a_val) / a_val) * 100.0
                    } else {
                        0.0
                    };
                    let arrow = if pct > 0.0 { "+" } else { "" };
                    out.push_str(&format!(
                        "  {key}: {a_val:.4} -> {b_val:.4} ({arrow}{pct:.1}%)\n"
                    ));
                }
                (Some(v), None) => out.push_str(&format!("  {key}: {v:.4} -> (missing)\n")),
                (None, Some(v)) => out.push_str(&format!("  {key}: (missing) -> {v:.4}\n")),
                (None, None) => {}
            }
        }
    }

    out
}
