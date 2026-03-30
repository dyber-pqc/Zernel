// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub status: ExperimentStatus,
    pub hyperparams: HashMap<String, serde_json::Value>,
    pub metrics: HashMap<String, f64>,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub git_commit: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum ExperimentStatus {
    Running,
    Done,
    Failed,
    Queued,
}

/// SQLite-backed experiment store at ~/.zernel/experiments/experiments.db
pub struct ExperimentStore {
    conn: Connection,
}

impl ExperimentStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                hyperparams TEXT NOT NULL,
                metrics TEXT NOT NULL,
                created_at TEXT NOT NULL,
                finished_at TEXT,
                git_commit TEXT
            );",
        )?;
        Ok(Self { conn })
    }

    pub fn insert(&self, exp: &Experiment) -> Result<()> {
        self.conn.execute(
            "INSERT INTO experiments (id, name, status, hyperparams, metrics, created_at, finished_at, git_commit)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            (
                &exp.id,
                &exp.name,
                serde_json::to_string(&exp.status)?,
                serde_json::to_string(&exp.hyperparams)?,
                serde_json::to_string(&exp.metrics)?,
                exp.created_at.to_rfc3339(),
                exp.finished_at.map(|t| t.to_rfc3339()),
                &exp.git_commit,
            ),
        )?;
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<Experiment>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, status, hyperparams, metrics, created_at, finished_at, git_commit FROM experiments ORDER BY created_at DESC")?;

        let experiments = stmt
            .query_map([], |row| {
                Ok(Experiment {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    status: serde_json::from_str(&row.get::<_, String>(2)?).unwrap_or(ExperimentStatus::Failed),
                    hyperparams: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or_default(),
                    metrics: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
                        .unwrap_or_default()
                        .with_timezone(&Utc),
                    finished_at: row
                        .get::<_, Option<String>>(6)?
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|t| t.with_timezone(&Utc)),
                    git_commit: row.get(7)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(experiments)
    }
}
