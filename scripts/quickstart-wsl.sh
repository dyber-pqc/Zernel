#!/usr/bin/env bash
# Zernel — WSL Quick Start
# Copyright (C) 2026 Dyber, Inc.
#
# Run this inside WSL (Ubuntu) to build and try everything.
set -euo pipefail

echo "======================================"
echo "  Zernel Quick Start (WSL)"
echo "======================================"
echo ""

# 1. Install Rust if needed
if ! command -v cargo &> /dev/null; then
    echo "[1/6] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[1/6] Rust already installed: $(rustc --version)"
fi

# 2. Build
echo "[2/6] Building Zernel (release mode)..."
cargo build --workspace --release

# 3. Add to PATH for this session
export PATH="$PWD/target/release:$PATH"

# 4. Run doctor
echo ""
echo "[3/6] Running zernel doctor..."
echo "--------------------------------------"
zernel doctor
echo ""

# 5. Demo: init + run + experiment tracking
echo "[4/6] Demo: creating a project and running a training script..."
echo "--------------------------------------"

# Create a demo project
mkdir -p /tmp/zernel-demo
cd /tmp/zernel-demo

cat > train_demo.py << 'PYEOF'
"""Zernel demo training script — generates fake metrics."""
import time
import math
import random

print("Starting training...")
print(f"Device: cpu (demo)")
print()

steps = 50
for step in range(1, steps + 1):
    loss = 2.0 * math.exp(-0.04 * step) + random.uniform(-0.05, 0.05)
    acc = 1.0 - loss / 2.5 + random.uniform(-0.02, 0.02)
    lr = 0.001 * (1.0 - step / steps)
    grad_norm = 0.5 + random.uniform(-0.1, 0.1)

    print(f"step: {step}/{steps}  loss: {loss:.4f}  accuracy: {acc:.4f}  lr: {lr:.6f}  grad_norm: {grad_norm:.4f}")
    time.sleep(0.05)

print()
print("Training complete!")
PYEOF

echo "Running: zernel run train_demo.py"
echo ""
zernel run train_demo.py || true
echo ""

# 6. Show experiments
echo "[5/6] Listing experiments..."
echo "--------------------------------------"
zernel exp list
echo ""

# 7. Query with ZQL
echo "[6/6] ZQL query..."
echo "--------------------------------------"
zernel query "SELECT * FROM experiments ORDER BY loss ASC"
echo ""

echo "======================================"
echo "  Try these next:"
echo "======================================"
echo ""
echo "  zernel watch              # Full-screen GPU dashboard (demo mode)"
echo "  zernel exp show <id>      # Show experiment details"
echo "  zernel model save <path>  # Save a model checkpoint"
echo "  zernel doctor             # Check your environment"
echo "  zerneld --simulate        # Start observability daemon"
echo "    then: curl localhost:9091/metrics"
echo ""
echo "  All data stored in: ~/.zernel/"
echo ""
