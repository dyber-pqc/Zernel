#!/usr/bin/env bash
# Zernel — Run All Tests
# Copyright (C) 2026 Dyber, Inc.
set -euo pipefail

echo "Running Zernel test suite..."

echo "=== Unit Tests ==="
cargo test --workspace

echo ""
echo "=== Clippy Lints ==="
cargo clippy --workspace -- -D warnings

echo ""
echo "=== Format Check ==="
cargo fmt --all -- --check

echo ""
echo "All checks passed."
