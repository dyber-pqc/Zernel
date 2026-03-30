# Zernel — AI-Native Linux OS
# Copyright (C) 2026 Dyber, Inc.
#
# Multi-stage build:
#   Stage 1: Build Rust binaries
#   Stage 2: Runtime image with NVIDIA CUDA base
#
# Build:  docker build -t zernel .
# Run:    docker run --gpus all -p 9091:9091 -p 9092:9092 zernel

# ============================================================
# Stage 1: Builder
# ============================================================
FROM rust:latest AS builder

WORKDIR /build

# Copy everything and build
COPY . .
RUN cargo build --workspace --release

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

LABEL org.opencontainers.image.title="Zernel"
LABEL org.opencontainers.image.description="AI-Native Linux OS for ML workloads"
LABEL org.opencontainers.image.vendor="Dyber, Inc."
LABEL org.opencontainers.image.source="https://github.com/dyber-pqc/Zernel"
LABEL org.opencontainers.image.licenses="GPL-2.0"

# Install Python and ML framework dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Zernel binaries
COPY --from=builder /build/target/release/zernel /usr/local/bin/zernel
COPY --from=builder /build/target/release/zernel-ebpf /usr/local/bin/zerneld
COPY --from=builder /build/target/release/zernel-scheduler /usr/local/bin/zernel-scheduler
COPY --from=builder /build/target/release/zernel-dashboard /usr/local/bin/zernel-dashboard

# Copy configuration
COPY distro/sysctl/99-zernel.conf /etc/sysctl.d/99-zernel.conf

# Create zernel data directory
RUN mkdir -p /etc/zernel /root/.zernel

# Ports: Prometheus metrics, WebSocket telemetry, Dashboard
EXPOSE 9091 9092 3000

# Default: run zerneld in simulate mode
# Override with: docker run zernel zernel-scheduler
ENTRYPOINT ["zerneld"]
CMD ["--simulate"]
