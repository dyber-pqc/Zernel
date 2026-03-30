// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub status: ExperimentStatus,
    pub hyperparams: HashMap<String, serde_json::Value>,
    pub metrics: HashMap<String, f64>,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub git_commit: Option<String>,
    pub script: Option<String>,
    pub duration_secs: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExperimentStatus {
    Running,
    Done,
    Failed,
    Queued,
}

impl std::fmt::Display for ExperimentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Running => write!(f, "Running"),
            Self::Done => write!(f, "Done"),
            Self::Failed => write!(f, "Failed"),
            Self::Queued => write!(f, "Queued"),
        }
    }
}

/// SQLite-backed experiment store.
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
                hyperparams TEXT NOT NULL DEFAULT '{}',
                metrics TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                finished_at TEXT,
                git_commit TEXT,
                script TEXT,
                duration_secs REAL
            );

            CREATE INDEX IF NOT EXISTS idx_experiments_created
                ON experiments(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_experiments_status
                ON experiments(status);",
        )?;
        Ok(Self { conn })
    }

    pub fn insert(&self, exp: &Experiment) -> Result<()> {
        self.conn.execute(
            "INSERT INTO experiments (id, name, status, hyperparams, metrics, created_at, finished_at, git_commit, script, duration_secs)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            (
                &exp.id,
                &exp.name,
                serde_json::to_string(&exp.status)?,
                serde_json::to_string(&exp.hyperparams)?,
                serde_json::to_string(&exp.metrics)?,
                exp.created_at.to_rfc3339(),
                exp.finished_at.map(|t| t.to_rfc3339()),
                &exp.git_commit,
                &exp.script,
                exp.duration_secs,
            ),
        )?;
        Ok(())
    }

    pub fn update_status(&self, id: &str, status: &ExperimentStatus) -> Result<()> {
        self.conn.execute(
            "UPDATE experiments SET status = ?1 WHERE id = ?2",
            (serde_json::to_string(status)?, id),
        )?;
        Ok(())
    }

    pub fn update_metrics(&self, id: &str, metrics: &HashMap<String, f64>) -> Result<()> {
        self.conn.execute(
            "UPDATE experiments SET metrics = ?1 WHERE id = ?2",
            (serde_json::to_string(metrics)?, id),
        )?;
        Ok(())
    }

    pub fn finish(&self, id: &str, status: ExperimentStatus, duration_secs: f64) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE experiments SET status = ?1, finished_at = ?2, duration_secs = ?3 WHERE id = ?4",
            (serde_json::to_string(&status)?, &now, duration_secs, id),
        )?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<Option<Experiment>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, status, hyperparams, metrics, created_at, finished_at, git_commit, script, duration_secs FROM experiments WHERE id = ?1"
        )?;

        let mut rows = stmt.query_map([id], Self::row_to_experiment)?;
        match rows.next() {
            Some(Ok(exp)) => Ok(Some(exp)),
            _ => Ok(None),
        }
    }

    pub fn list(&self, limit: usize) -> Result<Vec<Experiment>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, status, hyperparams, metrics, created_at, finished_at, git_commit, script, duration_secs FROM experiments ORDER BY created_at DESC LIMIT ?1"
        )?;

        let experiments = stmt
            .query_map([limit], Self::row_to_experiment)?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(experiments)
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        let count = self
            .conn
            .execute("DELETE FROM experiments WHERE id = ?1", [id])?;
        Ok(count > 0)
    }

    fn row_to_experiment(row: &rusqlite::Row) -> rusqlite::Result<Experiment> {
        Ok(Experiment {
            id: row.get(0)?,
            name: row.get(1)?,
            status: serde_json::from_str(&row.get::<_, String>(2)?)
                .unwrap_or(ExperimentStatus::Failed),
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
            script: row.get(8)?,
            duration_secs: row.get(9)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_store() -> (ExperimentStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let store = ExperimentStore::open(&path).unwrap();
        (store, dir)
    }

    fn make_experiment(id: &str) -> Experiment {
        Experiment {
            id: id.into(),
            name: format!("test-{id}"),
            status: ExperimentStatus::Running,
            hyperparams: HashMap::from([("lr".into(), serde_json::json!(0.001))]),
            metrics: HashMap::from([("loss".into(), 1.5)]),
            created_at: Utc::now(),
            finished_at: None,
            git_commit: Some("abc123".into()),
            script: Some("train.py".into()),
            duration_secs: None,
        }
    }

    #[test]
    fn insert_and_list() {
        let (store, _dir) = temp_store();
        store.insert(&make_experiment("exp-001")).unwrap();
        store.insert(&make_experiment("exp-002")).unwrap();

        let list = store.list(10).unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn get_by_id() {
        let (store, _dir) = temp_store();
        store.insert(&make_experiment("exp-001")).unwrap();

        let exp = store.get("exp-001").unwrap().unwrap();
        assert_eq!(exp.name, "test-exp-001");
    }

    #[test]
    fn finish_updates_status() {
        let (store, _dir) = temp_store();
        store.insert(&make_experiment("exp-001")).unwrap();
        store
            .finish("exp-001", ExperimentStatus::Done, 123.4)
            .unwrap();

        let exp = store.get("exp-001").unwrap().unwrap();
        assert_eq!(exp.status, ExperimentStatus::Done);
        assert!((exp.duration_secs.unwrap() - 123.4).abs() < 0.01);
    }

    #[test]
    fn delete_experiment() {
        let (store, _dir) = temp_store();
        store.insert(&make_experiment("exp-001")).unwrap();
        assert!(store.delete("exp-001").unwrap());
        assert!(store.get("exp-001").unwrap().is_none());
    }
}
