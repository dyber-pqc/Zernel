// Copyright (C) 2026 Dyber, Inc. — Proprietary

use crate::experiments::store::ExperimentStore;
use crate::experiments::tracker;
use crate::zql::parser::{self, CompareOp, Direction, Value, ZqlQuery};
use anyhow::Result;
use std::collections::HashMap;

/// Execute a ZQL query and return formatted results.
pub fn execute(query: &str) -> Result<String> {
    let ast = match parser::parse(query) {
        Ok(ast) => ast,
        Err(e) => return Ok(format!("Parse error: {e}")),
    };

    match ast.from.as_str() {
        "experiments" => execute_experiments(&ast),
        "jobs" => execute_jobs(&ast),
        "models" => execute_models(&ast),
        other => Ok(format!(
            "Unknown table: '{other}'. Available: experiments, jobs, models"
        )),
    }
}

/// Execute against the experiments SQLite table.
fn execute_experiments(ast: &ZqlQuery) -> Result<String> {
    let db_path = tracker::experiments_db_path();
    let store = ExperimentStore::open(&db_path)?;
    let all = store.list(10000)?;

    let rows: Vec<HashMap<String, String>> = all
        .into_iter()
        .map(|exp| {
            let mut row = HashMap::new();
            row.insert("id".into(), exp.id.clone());
            row.insert("name".into(), exp.name.clone());
            row.insert("status".into(), exp.status.to_string());
            row.insert("script".into(), exp.script.clone().unwrap_or_default());
            row.insert(
                "duration".into(),
                exp.duration_secs
                    .map(|d| format!("{d:.1}s"))
                    .unwrap_or_default(),
            );
            for (k, v) in &exp.metrics {
                row.insert(k.clone(), format!("{v:.4}"));
            }
            row
        })
        .collect();

    format_query_results(ast, rows)
}

/// Execute against the jobs SQLite table.
fn execute_jobs(ast: &ZqlQuery) -> Result<String> {
    let db_path = tracker::zernel_dir().join("jobs").join("jobs.db");
    if !db_path.exists() {
        return Ok("No jobs database found. Submit a job with: zernel job submit <script>".into());
    }

    let conn = rusqlite::Connection::open(&db_path)?;
    let mut stmt = conn.prepare(
        "SELECT id, script, status, gpus_per_node, nodes, framework, backend, exit_code, submitted_at FROM jobs ORDER BY submitted_at DESC",
    )?;

    let rows: Vec<HashMap<String, String>> = stmt
        .query_map([], |row| {
            let mut map = HashMap::new();
            map.insert("id".into(), row.get::<_, String>(0)?);
            map.insert("script".into(), row.get::<_, String>(1)?);
            map.insert("status".into(), row.get::<_, String>(2)?);
            map.insert("gpus_per_node".into(), row.get::<_, u32>(3)?.to_string());
            map.insert("nodes".into(), row.get::<_, u32>(4)?.to_string());
            map.insert("framework".into(), row.get::<_, String>(5)?);
            map.insert("backend".into(), row.get::<_, String>(6)?);
            map.insert(
                "exit_code".into(),
                row.get::<_, Option<i32>>(7)?
                    .map(|e| e.to_string())
                    .unwrap_or_default(),
            );
            map.insert("submitted_at".into(), row.get::<_, String>(8)?);
            Ok(map)
        })?
        .filter_map(|r| r.ok())
        .collect();

    format_query_results(ast, rows)
}

/// Execute against the models JSON registry.
fn execute_models(ast: &ZqlQuery) -> Result<String> {
    let registry_path = tracker::zernel_dir().join("models").join("registry.json");
    if !registry_path.exists() {
        return Ok("No model registry found. Save a model with: zernel model save <path>".into());
    }

    let data = std::fs::read_to_string(&registry_path)?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&data).unwrap_or_default();

    let rows: Vec<HashMap<String, String>> = entries
        .into_iter()
        .map(|entry| {
            let mut map = HashMap::new();
            if let Some(obj) = entry.as_object() {
                for (k, v) in obj {
                    map.insert(
                        k.clone(),
                        match v {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Number(n) => n.to_string(),
                            other => other.to_string(),
                        },
                    );
                }
            }
            map
        })
        .collect();

    format_query_results(ast, rows)
}

/// Generic query executor: applies WHERE, ORDER BY, LIMIT, then formats output.
fn format_query_results(ast: &ZqlQuery, rows: Vec<HashMap<String, String>>) -> Result<String> {
    // Apply WHERE filter
    let mut filtered: Vec<_> = rows
        .into_iter()
        .filter(|row| {
            if let Some(ref wc) = ast.where_clause {
                wc.conditions.iter().all(|cond| {
                    let field_val = row.get(&cond.field);
                    let Some(fv) = field_val else {
                        return false;
                    };

                    match &cond.value {
                        Value::Number(target) => {
                            let Ok(v) = fv.parse::<f64>() else {
                                return false;
                            };
                            match cond.op {
                                CompareOp::Eq => (v - target).abs() < f64::EPSILON,
                                CompareOp::NotEq => (v - target).abs() > f64::EPSILON,
                                CompareOp::Lt => v < *target,
                                CompareOp::Gt => v > *target,
                                CompareOp::Lte => v <= *target,
                                CompareOp::Gte => v >= *target,
                            }
                        }
                        Value::Text(target) => match cond.op {
                            CompareOp::Eq => fv == target,
                            CompareOp::NotEq => fv != target,
                            _ => false,
                        },
                    }
                })
            } else {
                true
            }
        })
        .collect();

    // Apply ORDER BY
    if let Some(ref ob) = ast.order_by {
        filtered.sort_by(|a, b| {
            let va = a
                .get(&ob.field)
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(f64::MAX);
            let vb = b
                .get(&ob.field)
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(f64::MAX);
            let cmp = va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal);
            match ob.direction {
                Direction::Asc => cmp,
                Direction::Desc => cmp.reverse(),
            }
        });
    }

    // Apply LIMIT
    if let Some(limit) = ast.limit {
        filtered.truncate(limit);
    }

    if filtered.is_empty() {
        return Ok("No results.".into());
    }

    // Determine columns
    let columns: Vec<&str> = if ast.select.len() == 1 && ast.select[0] == "*" {
        // Auto-detect columns from first row
        let mut cols: Vec<&str> = filtered[0].keys().map(|s| s.as_str()).collect();
        cols.sort();
        cols
    } else {
        ast.select.iter().map(|s| s.as_str()).collect()
    };

    // Format output
    let mut out = String::new();
    for col in &columns {
        out.push_str(&format!("{:<24}", col));
    }
    out.push('\n');
    out.push_str(&"-".repeat(columns.len() * 24));
    out.push('\n');

    for row in &filtered {
        for col in &columns {
            let val = row.get(*col).map(|s| s.as_str()).unwrap_or("-");
            out.push_str(&format!("{:<24}", val));
        }
        out.push('\n');
    }

    out.push_str(&format!("\n{} row(s)", filtered.len()));

    Ok(out)
}
