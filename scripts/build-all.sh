#!/usr/bin/env bash
# Zernel — Build All Components
# Copyright (C) 2026 Dyber, Inc.
set -euo pipefail

echo "Building Zernel workspace..."

cargo build --workspace --release

echo ""
echo "Build complete. Binaries:"
echo "  target/release/zernel-scheduler"
echo "  target/release/zernel-ebpf"
echo "  target/release/zernel"
