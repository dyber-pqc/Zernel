// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::Result;

/// Execute a ZQL query and return formatted results.
pub fn execute(query: &str) -> Result<String> {
    match super::parser::parse(query) {
        Ok(_ast) => {
            // TODO: Execute against SQLite experiment store and/or telemetry data
            Ok("(query execution not yet implemented)".into())
        }
        Err(e) => Ok(format!("Parse error: {e}")),
    }
}
