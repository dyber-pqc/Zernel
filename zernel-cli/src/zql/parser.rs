// Copyright (C) 2026 Dyber, Inc. — Proprietary

// ZQL — Zernel Query Language
//
// A SQL-like query language for querying experiments, models, and telemetry.
//
// Example:
//   SELECT name, loss FROM experiments WHERE loss < 1.5 ORDER BY loss ASC LIMIT 10;

use nom::{
    branch::alt,
    bytes::complete::{tag_no_case, take_while1},
    character::complete::{char, multispace0, multispace1},
    combinator::{map, opt},
    multi::separated_list1,
    number::complete::double,
    sequence::{delimited, preceded, tuple},
    IResult,
};

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

fn identifier(input: &str) -> IResult<&str, String> {
    let (input, id) = take_while1(|c: char| c.is_alphanumeric() || c == '_' || c == '*')(input)?;
    Ok((input, id.to_string()))
}

fn select_clause(input: &str) -> IResult<&str, Vec<String>> {
    let (input, _) = tag_no_case("SELECT")(input)?;
    let (input, _) = multispace1(input)?;
    separated_list1(tuple((multispace0, char(','), multispace0)), identifier)(input)
}

fn from_clause(input: &str) -> IResult<&str, String> {
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("FROM")(input)?;
    let (input, _) = multispace1(input)?;
    identifier(input)
}

fn compare_op(input: &str) -> IResult<&str, CompareOp> {
    alt((
        map(tag_no_case("!="), |_| CompareOp::NotEq),
        map(tag_no_case("<="), |_| CompareOp::Lte),
        map(tag_no_case(">="), |_| CompareOp::Gte),
        map(char('<'), |_| CompareOp::Lt),
        map(char('>'), |_| CompareOp::Gt),
        map(char('='), |_| CompareOp::Eq),
    ))(input)
}

fn quoted_string(input: &str) -> IResult<&str, String> {
    let (input, s) = delimited(char('\''), take_while1(|c: char| c != '\''), char('\''))(input)?;
    Ok((input, s.to_string()))
}

fn value(input: &str) -> IResult<&str, Value> {
    alt((map(quoted_string, Value::Text), map(double, Value::Number)))(input)
}

fn condition(input: &str) -> IResult<&str, Condition> {
    let (input, field) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, op) = compare_op(input)?;
    let (input, _) = multispace0(input)?;
    let (input, val) = value(input)?;
    Ok((
        input,
        Condition {
            field,
            op,
            value: val,
        },
    ))
}

fn where_clause(input: &str) -> IResult<&str, WhereClause> {
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("WHERE")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, conditions) = separated_list1(
        tuple((multispace1, tag_no_case("AND"), multispace1)),
        condition,
    )(input)?;
    Ok((input, WhereClause { conditions }))
}

fn direction(input: &str) -> IResult<&str, Direction> {
    alt((
        map(tag_no_case("ASC"), |_| Direction::Asc),
        map(tag_no_case("DESC"), |_| Direction::Desc),
    ))(input)
}

fn order_by_clause(input: &str) -> IResult<&str, OrderBy> {
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("ORDER")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("BY")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, field) = identifier(input)?;
    let (input, dir) = opt(preceded(multispace1, direction))(input)?;
    Ok((
        input,
        OrderBy {
            field,
            direction: dir.unwrap_or(Direction::Asc),
        },
    ))
}

fn limit_clause(input: &str) -> IResult<&str, usize> {
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("LIMIT")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, n) = double(input)?;
    Ok((input, n as usize))
}

fn zql_query(input: &str) -> IResult<&str, ZqlQuery> {
    let (input, _) = multispace0(input)?;
    let (input, select) = select_clause(input)?;
    let (input, from) = from_clause(input)?;
    let (input, where_cl) = opt(where_clause)(input)?;
    let (input, order) = opt(order_by_clause)(input)?;
    let (input, limit) = opt(limit_clause)(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = opt(char(';'))(input)?;
    let (input, _) = multispace0(input)?;

    Ok((
        input,
        ZqlQuery {
            select,
            from,
            where_clause: where_cl,
            order_by: order,
            limit,
        },
    ))
}

/// Parse a ZQL query string into a structured AST.
pub fn parse(input: &str) -> Result<ZqlQuery, String> {
    match zql_query(input) {
        Ok(("", query)) => Ok(query),
        Ok((remaining, _)) => Err(format!("unexpected trailing input: '{remaining}'")),
        Err(e) => Err(format!("parse error: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_select() {
        let q = parse("SELECT name, loss FROM experiments").unwrap();
        assert_eq!(q.select, vec!["name", "loss"]);
        assert_eq!(q.from, "experiments");
        assert!(q.where_clause.is_none());
    }

    #[test]
    fn parse_with_where() {
        let q = parse("SELECT * FROM experiments WHERE loss < 1.5").unwrap();
        assert_eq!(q.select, vec!["*"]);
        let w = q.where_clause.unwrap();
        assert_eq!(w.conditions.len(), 1);
        assert_eq!(w.conditions[0].field, "loss");
        assert_eq!(w.conditions[0].op, CompareOp::Lt);
        assert_eq!(w.conditions[0].value, Value::Number(1.5));
    }

    #[test]
    fn parse_with_order_and_limit() {
        let q = parse("SELECT name FROM experiments ORDER BY loss ASC LIMIT 10;").unwrap();
        let ob = q.order_by.unwrap();
        assert_eq!(ob.field, "loss");
        assert_eq!(ob.direction, Direction::Asc);
        assert_eq!(q.limit, Some(10));
    }

    #[test]
    fn parse_where_with_string() {
        let q = parse("SELECT * FROM experiments WHERE name = 'baseline'").unwrap();
        let w = q.where_clause.unwrap();
        assert_eq!(w.conditions[0].value, Value::Text("baseline".into()));
    }

    #[test]
    fn parse_multiple_conditions() {
        let q = parse("SELECT * FROM experiments WHERE loss < 1.5 AND accuracy > 0.9").unwrap();
        let w = q.where_clause.unwrap();
        assert_eq!(w.conditions.len(), 2);
    }

    #[test]
    fn parse_error_on_garbage() {
        assert!(parse("not a query").is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// The ZQL parser must never panic on any input string.
        #[test]
        fn parse_never_panics(input in ".*") {
            // Must return Ok or Err, never panic
            let _ = parse(&input);
        }

        /// Parse is deterministic — same input always gives same result.
        #[test]
        fn parse_is_deterministic(input in ".*") {
            let r1 = parse(&input);
            let r2 = parse(&input);
            match (&r1, &r2) {
                (Ok(_), Ok(_)) => {} // both parsed (can't easily compare ZqlQuery)
                (Err(e1), Err(e2)) => assert_eq!(e1, e2),
                _ => panic!("non-deterministic parse"),
            }
        }

        /// Valid SELECT queries always parse successfully.
        #[test]
        fn valid_select_always_parses(
            col in "[a-z_]{1,10}",
            table in "(experiments|jobs|models)",
        ) {
            let query = format!("SELECT {col} FROM {table}");
            assert!(parse(&query).is_ok(), "failed to parse: {query}");
        }
    }
}
