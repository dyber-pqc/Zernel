#!/usr/bin/env bash
# Zernel ISO Builder
# Copyright (C) 2026 Dyber, Inc.
# Requires: live-build, debootstrap, squashfs-tools
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
ISO_NAME="zernel-0.1.0-amd64.iso"

echo "Building Zernel ISO..."
echo "  Repository: $REPO_ROOT"
echo "  Build dir:  $BUILD_DIR"
echo "  Output:     $SCRIPT_DIR/$ISO_NAME"

# TODO: Implement ISO build using live-build
# Steps:
# 1. lb config (configure live-build)
# 2. Add Zernel kernel package
# 3. Add NVIDIA packages
# 4. Add Zernel tools (scheduler, zerneld, CLI)
# 5. Add sysctl configuration
# 6. Add installer
# 7. lb build

echo "ISO build not yet implemented."
echo "Prerequisites: apt install live-build debootstrap squashfs-tools"
