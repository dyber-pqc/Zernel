// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

//! Predictive Data Prefetching
//!
//! Uses phase timing data from the sched_ext scheduler to predict when
//! the GpuCompute phase will end and signal the DataLoader to start
//! prefetching the next batch BEFORE the GPU finishes.
//!
//! This eliminates the "GPU idle waiting for data" gap that causes
//! 5-30% throughput loss in data-bound training.

use std::collections::VecDeque;
use tracing::debug;

/// Tracks phase durations to predict when to trigger prefetch.
pub struct PrefetchPredictor {
    /// Recent GpuCompute phase durations (ns).
    compute_durations: VecDeque<u64>,
    /// Recent DataLoading phase durations (ns).
    loading_durations: VecDeque<u64>,
    /// Maximum history size.
    max_history: usize,
    /// How far before predicted compute end to trigger prefetch (ns).
    prefetch_lead_ns: u64,
}

impl PrefetchPredictor {
    pub fn new(max_history: usize, prefetch_lead_ns: u64) -> Self {
        Self {
            compute_durations: VecDeque::with_capacity(max_history),
            loading_durations: VecDeque::with_capacity(max_history),
            max_history,
            prefetch_lead_ns,
        }
    }

    /// Record a completed GpuCompute phase duration.
    pub fn record_compute(&mut self, duration_ns: u64) {
        if self.compute_durations.len() >= self.max_history {
            self.compute_durations.pop_front();
        }
        self.compute_durations.push_back(duration_ns);
    }

    /// Record a completed DataLoading phase duration.
    pub fn record_loading(&mut self, duration_ns: u64) {
        if self.loading_durations.len() >= self.max_history {
            self.loading_durations.pop_front();
        }
        self.loading_durations.push_back(duration_ns);
    }

    /// Predict the next GpuCompute duration using exponential moving average.
    pub fn predicted_compute_ns(&self) -> Option<u64> {
        if self.compute_durations.is_empty() {
            return None;
        }
        // Exponential moving average (alpha = 0.3)
        let alpha = 0.3;
        let mut ema = self.compute_durations[0] as f64;
        for &d in self.compute_durations.iter().skip(1) {
            ema = alpha * d as f64 + (1.0 - alpha) * ema;
        }
        Some(ema as u64)
    }

    /// Predict the next DataLoading duration.
    pub fn predicted_loading_ns(&self) -> Option<u64> {
        if self.loading_durations.is_empty() {
            return None;
        }
        let alpha = 0.3;
        let mut ema = self.loading_durations[0] as f64;
        for &d in self.loading_durations.iter().skip(1) {
            ema = alpha * d as f64 + (1.0 - alpha) * ema;
        }
        Some(ema as u64)
    }

    /// Should we trigger prefetch now?
    /// Call this periodically during GpuCompute phase with elapsed time.
    pub fn should_prefetch(&self, elapsed_compute_ns: u64) -> bool {
        let Some(predicted) = self.predicted_compute_ns() else {
            return false;
        };

        if elapsed_compute_ns + self.prefetch_lead_ns >= predicted {
            debug!(
                elapsed_ms = elapsed_compute_ns / 1_000_000,
                predicted_ms = predicted / 1_000_000,
                lead_ms = self.prefetch_lead_ns / 1_000_000,
                "triggering predictive prefetch"
            );
            return true;
        }

        false
    }

    /// Calculate the overlap efficiency.
    /// 1.0 = perfect overlap (data ready exactly when GPU finishes)
    /// <1.0 = GPU had to wait for data
    /// >1.0 = data was ready before GPU finished (ideal)
    pub fn overlap_efficiency(&self) -> f64 {
        let compute = self.predicted_compute_ns().unwrap_or(1) as f64;
        let loading = self.predicted_loading_ns().unwrap_or(1) as f64;
        if loading == 0.0 {
            return 1.0;
        }
        (compute + self.prefetch_lead_ns as f64) / loading
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_converges() {
        let mut p = PrefetchPredictor::new(10, 5_000_000); // 5ms lead
                                                           // Simulate consistent 100ms compute phases
        for _ in 0..10 {
            p.record_compute(100_000_000);
        }
        let predicted = p.predicted_compute_ns().unwrap();
        assert!((predicted as f64 - 100_000_000.0).abs() < 1_000_000.0);
    }

    #[test]
    fn prefetch_triggers_near_end() {
        let mut p = PrefetchPredictor::new(10, 10_000_000); // 10ms lead
        for _ in 0..5 {
            p.record_compute(100_000_000); // 100ms phases
        }
        // At 85ms into compute (15ms before predicted end, within 10ms lead)
        assert!(!p.should_prefetch(80_000_000)); // too early
        assert!(p.should_prefetch(95_000_000)); // within lead time
    }

    #[test]
    fn overlap_efficiency_calculation() {
        let mut p = PrefetchPredictor::new(10, 10_000_000);
        for _ in 0..5 {
            p.record_compute(100_000_000);
            p.record_loading(50_000_000);
        }
        // compute(100ms) + lead(10ms) / loading(50ms) = 2.2
        let eff = p.overlap_efficiency();
        assert!(eff > 1.0); // data ready well before GPU finishes
    }
}
