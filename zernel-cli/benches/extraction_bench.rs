// Copyright (C) 2026 Dyber, Inc. — Proprietary
//
// Benchmarks for metric extraction and ZQL parsing.

use criterion::{criterion_group, criterion_main, Criterion};
use regex::Regex;

fn bench_metric_extraction(c: &mut Criterion) {
    // Pre-compile regexes (matching MetricExtractor::new())
    let patterns: Vec<(&str, Regex)> = vec![
        (
            "loss",
            Regex::new(r"(?i)\bloss[=:\s]+([0-9]+\.?[0-9]*)").unwrap(),
        ),
        (
            "accuracy",
            Regex::new(r"(?i)\b(?:accuracy|acc)[=:\s]+([0-9]+\.?[0-9]*)").unwrap(),
        ),
        (
            "grad_norm",
            Regex::new(r"(?i)\bgrad_norm[=:\s]+([0-9]+\.?[0-9]*)").unwrap(),
        ),
        (
            "lr",
            Regex::new(r"(?i)\b(?:learning_rate|lr)[=:\s]+([0-9]+\.?[0-9eE\-]*)").unwrap(),
        ),
        (
            "step",
            Regex::new(r"(?i)\b(?:step|global_step)[=:\s]+([0-9]+)").unwrap(),
        ),
    ];

    let lines: Vec<String> = (0..1000)
        .map(|i| {
            format!(
                "step: {i}/10000 loss: {:.4} accuracy: {:.4} lr: {:.6} grad_norm: {:.4}",
                2.0 * (-0.001 * i as f64).exp(),
                0.5 + 0.3 * (i as f64 / 1000.0),
                0.001 * (1.0 - i as f64 / 10000.0),
                0.5 + 0.1 * (i as f64 * 0.01).sin()
            )
        })
        .collect();

    c.bench_function("metric_extraction_1000_lines", |b| {
        b.iter(|| {
            let mut total_extracted = 0usize;
            for line in &lines {
                for (_, regex) in &patterns {
                    if regex.captures(line).is_some() {
                        total_extracted += 1;
                    }
                }
            }
            total_extracted
        });
    });
}

fn bench_zql_parse(c: &mut Criterion) {
    use nom::{
        branch::alt,
        bytes::complete::{tag_no_case, take_while1},
        character::complete::{char, multispace0, multispace1},
        combinator::map,
        multi::separated_list1,
        sequence::tuple,
        IResult,
    };

    fn identifier(input: &str) -> IResult<&str, String> {
        let (input, id) =
            take_while1(|c: char| c.is_alphanumeric() || c == '_' || c == '*')(input)?;
        Ok((input, id.to_string()))
    }

    fn select_clause(input: &str) -> IResult<&str, Vec<String>> {
        let (input, _) = tag_no_case("SELECT")(input)?;
        let (input, _) = multispace1(input)?;
        separated_list1(tuple((multispace0, char(','), multispace0)), identifier)(input)
    }

    let queries = vec![
        "SELECT name, loss FROM experiments WHERE loss < 1.5 ORDER BY loss ASC LIMIT 10",
        "SELECT * FROM experiments",
        "SELECT name, accuracy FROM experiments WHERE accuracy > 0.9 AND loss < 0.5",
        "SELECT id, status FROM jobs WHERE status = 'running'",
    ];

    c.bench_function("zql_parse_1000_queries", |b| {
        b.iter(|| {
            let mut parsed = 0;
            for _ in 0..250 {
                for q in &queries {
                    if select_clause(q).is_ok() {
                        parsed += 1;
                    }
                }
            }
            parsed
        });
    });
}

criterion_group!(benches, bench_metric_extraction, bench_zql_parse);
criterion_main!(benches);
