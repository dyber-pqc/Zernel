// Copyright (C) 2026 Dyber, Inc. — GPL-2.0
//
// Benchmarks for the Zernel ML scheduler.

use criterion::{criterion_group, criterion_main, Criterion};

// We can't import from the binary crate directly.
// These benchmarks test the core algorithms using inline implementations
// that mirror the production code.

fn bench_phase_detection(c: &mut Criterion) {
    c.bench_function("phase_detection_1000_tasks", |b| {
        b.iter(|| {
            // Simulate phase detection for 1000 tasks
            let mut results = Vec::with_capacity(1000);
            for i in 0..1000u32 {
                let gpu_util = (i * 7 % 100) as u8;
                let io_wait = (i * 13 % 100) as f32 / 100.0;
                let phase = if io_wait > 0.3 && gpu_util < 10 {
                    "DataLoading"
                } else if gpu_util > 80 {
                    "GpuCompute"
                } else {
                    "Unknown"
                };
                results.push(phase);
            }
            results
        });
    });
}

fn bench_scheduling_decision(c: &mut Criterion) {
    c.bench_function("scheduling_decision_10000", |b| {
        b.iter(|| {
            let mut total_priority = 0i64;
            for i in 0..10000u32 {
                let phase = i % 5;
                let priority: i32 = match phase {
                    0 => 10, // DataLoading
                    1 => -5, // GpuCompute
                    2 => 10, // NcclCollective
                    3 => 10, // OptimizerStep
                    _ => 0,  // Unknown
                };
                total_priority += priority as i64;
            }
            total_priority
        });
    });
}

fn bench_numa_cpu_selection(c: &mut Criterion) {
    use std::collections::HashMap;

    c.bench_function("numa_cpu_selection_64_cpus", |b| {
        let loads: HashMap<u32, f32> = (0..64).map(|i| (i, (i as f32 * 0.7) % 1.0)).collect();
        let preferred: Vec<u32> = (0..32).collect();

        b.iter(|| {
            preferred
                .iter()
                .copied()
                .min_by(|a, b| {
                    let la = loads.get(a).unwrap_or(&0.0);
                    let lb = loads.get(b).unwrap_or(&0.0);
                    la.partial_cmp(lb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0)
        });
    });
}

criterion_group!(
    benches,
    bench_phase_detection,
    bench_scheduling_decision,
    bench_numa_cpu_selection
);
criterion_main!(benches);
