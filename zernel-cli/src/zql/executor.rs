// Copyright (C) 2026 Dyber, Inc. — Proprietary

use crate::experiments::store::ExperimentStore;
use crate::experiments::tracker;
use crate::zql::parser::{self, CompareOp, Direction, Value};
use anyhow::Result;

/// Execute a ZQL query and return formatted results.
pub fn execute(query: &str) -> Result<String> {
    let ast = match parser::parse(query) {
        Ok(ast) => ast,
        Err(e) => return Ok(format!("Parse error: {e}")),
    };

    // Only experiments table is supported for now
    if ast.from != "experiments" {
        return Ok(format!(
            "Unknown table: '{}'. Available tables: experiments",
            ast.from
        ));
    }

    let db_path = tracker::experiments_db_path();
    let store = ExperimentStore::open(&db_path)?;
    let all = store.list(1000)?;

    // Apply WHERE filter
    let mut results: Vec<_> = all
        .into_iter()
        .filter(|exp| {
            if let Some(ref wc) = ast.where_clause {
                wc.conditions.iter().all(|cond| {
                    let field_val = match cond.field.as_str() {
                        "name" => Some(Value::Text(exp.name.clone())),
                        "id" => Some(Value::Text(exp.id.clone())),
                        "status" => Some(Value::Text(exp.status.to_string())),
                        "loss" => exp.metrics.get("loss").map(|v| Value::Number(*v)),
                        "accuracy" => exp.metrics.get("accuracy").map(|v| Value::Number(*v)),
                        "learning_rate" => {
                            exp.metrics.get("learning_rate").map(|v| Value::Number(*v))
                        }
                        other => exp.metrics.get(other).map(|v| Value::Number(*v)),
                    };

                    let Some(fv) = field_val else {
                        return false;
                    };

                    match (&fv, &cond.value) {
                        (Value::Number(a), Value::Number(b)) => match cond.op {
                            CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                            CompareOp::NotEq => (a - b).abs() > f64::EPSILON,
                            CompareOp::Lt => a < b,
                            CompareOp::Gt => a > b,
                            CompareOp::Lte => a <= b,
                            CompareOp::Gte => a >= b,
                        },
                        (Value::Text(a), Value::Text(b)) => match cond.op {
                            CompareOp::Eq => a == b,
                            CompareOp::NotEq => a != b,
                            _ => false,
                        },
                        _ => false,
                    }
                })
            } else {
                true
            }
        })
        .collect();

    // Apply ORDER BY
    if let Some(ref ob) = ast.order_by {
        results.sort_by(|a, b| {
            let va = get_sortable_value(a, &ob.field);
            let vb = get_sortable_value(b, &ob.field);
            let cmp = va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal);
            match ob.direction {
                Direction::Asc => cmp,
                Direction::Desc => cmp.reverse(),
            }
        });
    }

    // Apply LIMIT
    if let Some(limit) = ast.limit {
        results.truncate(limit);
    }

    if results.is_empty() {
        return Ok("No results.".into());
    }

    // Format output
    let columns = &ast.select;
    let is_star = columns.len() == 1 && columns[0] == "*";
    let display_cols: Vec<&str> = if is_star {
        vec!["id", "name", "status", "loss", "accuracy"]
    } else {
        columns.iter().map(|s| s.as_str()).collect()
    };

    // Header
    let mut out = String::new();
    for col in &display_cols {
        out.push_str(&format!("{:<24}", col));
    }
    out.push('\n');
    out.push_str(&"-".repeat(display_cols.len() * 24));
    out.push('\n');

    // Rows
    for exp in &results {
        for col in &display_cols {
            let val = match *col {
                "id" => exp.id.clone(),
                "name" => exp.name.clone(),
                "status" => exp.status.to_string(),
                "script" => exp.script.clone().unwrap_or_default(),
                other => exp
                    .metrics
                    .get(other)
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "-".into()),
            };
            out.push_str(&format!("{:<24}", val));
        }
        out.push('\n');
    }

    out.push_str(&format!("\n{} row(s)", results.len()));

    Ok(out)
}

fn get_sortable_value(exp: &crate::experiments::store::Experiment, field: &str) -> f64 {
    match field {
        "name" => 0.0, // string sort not supported for f64, but works for basic ordering
        _ => exp.metrics.get(field).copied().unwrap_or(f64::MAX),
    }
}
