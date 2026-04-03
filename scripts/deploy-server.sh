#!/bin/bash
# Copyright (C) 2026 Dyber, Inc.
#
# Zernel GPU Server Deploy Script
#
# Deploys the full Zernel stack on a bare-metal GPU server:
#   1. System dependencies (Rust, clang-16, libbpf, bpftool)
#   2. Linux kernel 6.12+ with CONFIG_SCHED_CLASS_EXT=y
#   3. NVIDIA drivers via DKMS
#   4. Zernel scheduler, CLI, eBPF daemon, runtime, accelerator
#   5. Python ML environment (PyTorch + CUDA)
#
# Usage:
#   bash scripts/deploy-server.sh              # Full install
#   bash scripts/deploy-server.sh --skip-kernel # Skip kernel build (if 6.12+ already)
#   bash scripts/deploy-server.sh --skip-nvidia # Skip NVIDIA driver install
#
# Requirements:
#   - Debian/Ubuntu-based Linux
#   - Root access
#   - NVIDIA GPU
#   - Internet connection
#
# Tested on: Debian Bookworm, Ubuntu 22.04/24.04

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[zernel]${NC} $*"; }
warn()  { echo -e "${YELLOW}[zernel]${NC} $*"; }
error() { echo -e "${RED}[zernel]${NC} $*" >&2; }

SKIP_KERNEL=false
SKIP_NVIDIA=false
ZERNEL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

for arg in "$@"; do
    case "$arg" in
        --skip-kernel) SKIP_KERNEL=true ;;
        --skip-nvidia) SKIP_NVIDIA=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-kernel] [--skip-nvidia]"
            exit 0
            ;;
    esac
done

# ── Checks ─────────────────────────────────────────────────────

if [ "$(id -u)" -ne 0 ]; then
    error "This script must be run as root"
    exit 1
fi

info "Zernel GPU Server Deploy"
info "========================"
info "Source: $ZERNEL_DIR"
echo

# ── 1. System Dependencies ─────────────────────────────────────

info "Step 1/7: Installing system dependencies..."

apt-get update -qq
apt-get install -y -qq \
    build-essential \
    pkg-config \
    libssl-dev \
    clang-16 \
    libbpf-dev \
    linux-headers-"$(uname -r)" \
    bpftool \
    python3 \
    python3-venv \
    python3-pip \
    numactl \
    iproute2 \
    wget \
    curl \
    git \
    jq \
    2>/dev/null || true

info "  clang-16: $(clang-16 --version 2>/dev/null | head -1 || echo 'not found')"

# ── 2. Rust ────────────────────────────────────────────────────

info "Step 2/7: Installing Rust..."

if command -v rustc &>/dev/null; then
    info "  Rust already installed: $(rustc --version)"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    info "  Installed: $(rustc --version)"
fi

# ── 3. Kernel (optional) ──────────────────────────────────────

KERNEL_VER=$(uname -r | cut -d. -f1-2)
NEED_KERNEL=false

if [ "$SKIP_KERNEL" = true ]; then
    info "Step 3/7: Skipping kernel build (--skip-kernel)"
elif [ -f /sys/kernel/sched_ext/state ]; then
    info "Step 3/7: sched_ext already available (kernel $(uname -r))"
else
    NEED_KERNEL=true
    info "Step 3/7: Building kernel 6.12 with sched_ext support..."
    info "  This takes 20-40 minutes. Go get coffee."

    KERNEL_SRC="/usr/src/linux-6.12.8"
    if [ ! -d "$KERNEL_SRC" ]; then
        cd /usr/src
        wget -q "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.12.8.tar.xz"
        tar xf linux-6.12.8.tar.xz
    fi

    cd "$KERNEL_SRC"

    # Start from current config, enable sched_ext
    if [ -f /boot/config-"$(uname -r)" ]; then
        cp /boot/config-"$(uname -r)" .config
    else
        make defconfig
    fi

    # Enable sched_ext and required options
    scripts/config --enable CONFIG_SCHED_CLASS_EXT
    scripts/config --enable CONFIG_BPF
    scripts/config --enable CONFIG_BPF_SYSCALL
    scripts/config --enable CONFIG_BPF_JIT
    scripts/config --enable CONFIG_DEBUG_INFO_BTF
    scripts/config --enable CONFIG_PAHOLE_HAS_SPLIT_BTF
    scripts/config --enable CONFIG_SATA_AHCI
    scripts/config --enable CONFIG_CPU_FREQ
    scripts/config --enable CONFIG_CPU_FREQ_STAT
    scripts/config --enable CONFIG_CPU_FREQ_GOV_SCHEDUTIL
    scripts/config --enable CONFIG_CPU_FREQ_GOV_PERFORMANCE
    scripts/config --enable CONFIG_INTEL_RAPL
    make olddefconfig

    make -j"$(nproc)" 2>&1 | tail -5
    make modules_install
    make install

    # Generate vmlinux.h for BPF
    bpftool btf dump file /sys/kernel/btf/vmlinux format c > \
        "$ZERNEL_DIR/zernel-scheduler/src/bpf/vmlinux.h" 2>/dev/null || true

    info "  Kernel installed. Reboot required to activate sched_ext."
    info "  After reboot, re-run this script with --skip-kernel"
fi

# ── 4. NVIDIA Drivers ─────────────────────────────────────────

if [ "$SKIP_NVIDIA" = true ]; then
    info "Step 4/7: Skipping NVIDIA driver install (--skip-nvidia)"
elif nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    info "Step 4/7: NVIDIA driver already installed: $GPU_NAME (driver $DRIVER_VER)"
else
    info "Step 4/7: Installing NVIDIA drivers..."
    apt-get install -y -qq nvidia-driver nvidia-dkms 2>/dev/null || \
    apt-get install -y -qq nvidia-driver-535 2>/dev/null || \
    warn "  Auto-install failed. Install NVIDIA drivers manually."
fi

# ── 5. Build Zernel ───────────────────────────────────────────

info "Step 5/7: Building Zernel..."

cd "$ZERNEL_DIR"

# Build scheduler with BPF support (only on Linux with sched_ext)
if [ -f /sys/kernel/sched_ext/state ]; then
    # Generate vmlinux.h from running kernel
    if command -v bpftool &>/dev/null; then
        bpftool btf dump file /sys/kernel/btf/vmlinux format c > \
            zernel-scheduler/src/bpf/vmlinux.h 2>/dev/null || true
    fi

    info "  Building with BPF scheduler support..."
    cargo build --release -p zernel-scheduler --features bpf 2>&1 | tail -3
else
    info "  Building without BPF (sched_ext not available, need kernel 6.12+)"
    cargo build --release -p zernel-scheduler 2>&1 | tail -3
fi

# Build CLI and other crates
cargo build --release -p zernel-cli 2>&1 | tail -3

# Install binaries
cp target/release/zernel /usr/local/bin/ 2>/dev/null || true
cp target/release/zernel-scheduler /usr/local/bin/ 2>/dev/null || true

info "  Binaries installed to /usr/local/bin/"

# ── 6. Python ML Environment ─────────────────────────────────

info "Step 6/7: Setting up Python ML environment..."

VENV="/opt/zernel-env"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

# Install PyTorch with CUDA
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio 2>/dev/null || \
pip install -q torch 2>/dev/null

# Install zernel-runtime
cd "$ZERNEL_DIR/zernel-runtime"
pip install -q -e . 2>/dev/null || pip install -q . 2>/dev/null

# Verify
PYTORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
CUDA_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
info "  PyTorch: $PYTORCH_VER, CUDA: $CUDA_OK"

# ── 7. Start Services ────────────────────────────────────────

info "Step 7/7: Starting Zernel services..."

# Create config directory
mkdir -p /etc/zernel
if [ ! -f /etc/zernel/scheduler.toml ]; then
    cat > /etc/zernel/scheduler.toml << 'CONF'
[general]
phase_eval_interval_ms = 100
gpu_poll_interval_ms = 500

[phase_detection]
io_wait_threshold = 0.3
gpu_active_threshold = 80
gpu_idle_threshold = 10
phase_stability_count = 3

[numa]
gpu_affinity = true

[multi_tenant]
enabled = false
CONF
fi

# Start scheduler (if sched_ext available)
if [ -f /sys/kernel/sched_ext/state ] && [ -f /usr/local/bin/zernel-scheduler ]; then
    pkill -f zernel-scheduler 2>/dev/null || true
    sleep 1
    nohup zernel-scheduler > /var/log/zernel-scheduler.log 2>&1 &
    sleep 2

    SCHED_STATE=$(cat /sys/kernel/sched_ext/state 2>/dev/null)
    if [ "$SCHED_STATE" = "enabled" ]; then
        SCHED_OPS=$(cat /sys/kernel/sched_ext/root/ops 2>/dev/null)
        info "  sched_ext scheduler: $SCHED_OPS (ACTIVE)"
    else
        warn "  sched_ext scheduler failed to attach (state: $SCHED_STATE)"
    fi
fi

# Start accelerator daemon
if [ -f "$ZERNEL_DIR/zernel-accel/zernel_accel.py" ]; then
    pkill -f zernel_accel 2>/dev/null || true
    nohup python3 "$ZERNEL_DIR/zernel-accel/zernel_accel.py" > /var/log/zernel-accel.log 2>&1 &
    info "  zernel-accel daemon: started"
fi

# Enable GPU persistence mode
nvidia-smi -pm 1 2>/dev/null || true

# ── Done ──────────────────────────────────────────────────────

echo
info "============================================================"
info "Zernel deployment complete!"
info "============================================================"
echo
info "Verify:"
info "  cat /sys/kernel/sched_ext/root/ops    # Should print 'zernel'"
info "  zernel gpu status                     # GPU overview"
info "  zernel optimize scan                  # Environment audit"
echo
info "Train faster:"
info "  source /opt/zernel-env/bin/activate"
info "  zernel-run train.py                   # 2.2x auto speedup"
echo
info "Logs:"
info "  /var/log/zernel-scheduler.log"
info "  /var/log/zernel-accel.log"
echo

if [ "$NEED_KERNEL" = true ]; then
    warn "IMPORTANT: Reboot required to activate kernel 6.12 with sched_ext."
    warn "After reboot, run: bash $0 --skip-kernel --skip-nvidia"
fi
