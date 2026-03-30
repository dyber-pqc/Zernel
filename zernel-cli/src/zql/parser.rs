// Copyright (C) 2026 Dyber, Inc. — Proprietary

/// ZQL — Zernel Query Language
///
/// A SQL-like query language for querying experiments, models, and telemetry.
///
/// Example:
///   SELECT name, loss, learning_rate FROM experiments WHERE loss < 1.5 ORDER BY loss ASC;

#[derive(Debug, PartialEq)]
pub struct ZqlQuery {
    pub select: Vec<String>,
    pub from: String,
    pub where_clause: Option<WhereClause>,
    pub order_by: Option<OrderBy>,
    pub limit: Option<usize>,
}

#[derive(Debug, PartialEq)]
pub struct WhereClause {
    pub conditions: Vec<Condition>,
}

#[derive(Debug, PartialEq)]
pub struct Condition {
    pub field: String,
    pub op: CompareOp,
    pub value: Value,
}

#[derive(Debug, PartialEq)]
pub enum CompareOp {
    Eq,
    NotEq,
    Lt,
    Gt,
    Lte,
    Gte,
}

#[derive(Debug, PartialEq)]
pub enum Value {
    Number(f64),
    Text(String),
}

#[derive(Debug, PartialEq)]
pub struct OrderBy {
    pub field: String,
    pub direction: Direction,
}

#[derive(Debug, PartialEq)]
pub enum Direction {
    Asc,
    Desc,
}

/// Parse a ZQL query string into a structured AST.
pub fn parse(_input: &str) -> Result<ZqlQuery, String> {
    // TODO: Implement with nom parser combinators
    // For now, return a helpful error
    Err("ZQL parser not yet implemented. Coming in Phase 3.".into())
}
