#!/usr/bin/env bash
# Zernel — Developer Environment Setup
# Copyright (C) 2026 Dyber, Inc.
set -euo pipefail

echo "Setting up Zernel development environment..."

# Rust toolchain
if ! command -v rustup &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

rustup toolchain install stable
rustup component add rustfmt clippy

# Verify
echo ""
echo "Toolchain:"
rustc --version
cargo --version

echo ""
echo "Building workspace..."
cargo build --workspace

echo ""
echo "Running tests..."
cargo test --workspace

echo ""
echo "Development environment ready."
