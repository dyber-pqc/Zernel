#!/usr/bin/env bash
# Zernel ISO Builder
# Copyright (C) 2026 Dyber, Inc.
#
# Builds a bootable Zernel Linux ISO using Debian live-build.
# Requires: live-build, debootstrap, squashfs-tools, sudo
#
# Usage: sudo ./build-iso.sh [--profile desktop|server] [--arch amd64|arm64]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
PROFILE="${ZERNEL_PROFILE:-server}"
ARCH="amd64"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile) PROFILE="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        *) shift ;;
    esac
done
VERSION="0.1.0"
ISO_NAME="zernel-${VERSION}-${PROFILE}-${ARCH}.iso"

echo "╔═══════════════════════════════════════╗"
echo "║     Zernel ISO Builder v${VERSION}          ║"
echo "╚═══════════════════════════════════════╝"
echo ""
echo "  Profile:      ${PROFILE}"
echo "  Architecture: ${ARCH}"
echo "  Repository:   ${REPO_ROOT}"
echo "  Build dir:    ${BUILD_DIR}"
echo "  Output:       ${SCRIPT_DIR}/${ISO_NAME}"
echo ""

# Check prerequisites
for cmd in lb debootstrap mksquashfs cargo; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "ERROR: ${cmd} not found."
        echo "Install: apt install live-build debootstrap squashfs-tools"
        exit 1
    fi
done

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Must run as root (sudo)."
    exit 1
fi

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ============================================================
# Step 1: Configure live-build
# ============================================================
echo "[1/7] Configuring live-build..."
lb config \
    --architecture "${ARCH}" \
    --distribution bookworm \
    --archive-areas "main contrib non-free non-free-firmware" \
    --bootloaders grub-efi \
    --binary-images iso-hybrid \
    --iso-application "Zernel" \
    --iso-publisher "Dyber, Inc." \
    --iso-volume "Zernel ${VERSION}" \
    --memtest none

# ============================================================
# Step 2: Add base packages
# ============================================================
echo "[2/8] Adding base packages..."
cp "$REPO_ROOT/distro/packages/base-packages.list" \
    config/package-lists/base.list.chroot
cp "$REPO_ROOT/distro/packages/ai-ml-stack.list" \
    config/package-lists/ai-ml.list.chroot

if [ "$PROFILE" = "desktop" ]; then
    echo "  Adding GNOME desktop packages..."
    cp "$REPO_ROOT/distro/packages/gnome-packages.list" \
        config/package-lists/gnome.list.chroot
    cp "$REPO_ROOT/distro/packages/desktop-apps.list" \
        config/package-lists/desktop-apps.list.chroot
fi

# ============================================================
# Step 3: NVIDIA repository and packages
# ============================================================
echo "[3/7] Adding NVIDIA repository..."
mkdir -p config/archives
cat > config/archives/nvidia.list.chroot << 'EOF'
deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /
EOF

cat > config/hooks/live/01-nvidia-keyring.hook.chroot << 'HOOK'
#!/bin/bash
apt-get update -qq
apt-get install -y --no-install-recommends gnupg2 curl
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-cuda.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" > /etc/apt/sources.list.d/nvidia-cuda.list
apt-get update -qq
HOOK
chmod +x config/hooks/live/01-nvidia-keyring.hook.chroot

# ============================================================
# Step 4: Build Zernel binaries
# ============================================================
echo "[4/7] Building Zernel binaries..."
(cd "$REPO_ROOT" && cargo build --workspace --release)

mkdir -p config/includes.chroot/usr/local/bin
cp "$REPO_ROOT/target/release/zernel" config/includes.chroot/usr/local/bin/
cp "$REPO_ROOT/target/release/zernel-ebpf" config/includes.chroot/usr/local/bin/zerneld
cp "$REPO_ROOT/target/release/zernel-scheduler" config/includes.chroot/usr/local/bin/
cp "$REPO_ROOT/target/release/zernel-dashboard" config/includes.chroot/usr/local/bin/ 2>/dev/null || true

# ML stack setup script and zernel install wrapper
mkdir -p config/includes.chroot/opt/zernel
cp "$REPO_ROOT/distro/scripts/setup-ml-stack.sh" config/includes.chroot/opt/zernel/
cp "$REPO_ROOT/distro/scripts/zernel-install" config/includes.chroot/usr/local/bin/
chmod +x config/includes.chroot/usr/local/bin/zernel-install

# ============================================================
# Step 5: Add sysctl configuration
# ============================================================
echo "[5/7] Adding sysctl configuration..."
mkdir -p config/includes.chroot/etc/sysctl.d
cp "$REPO_ROOT/distro/sysctl/99-zernel.conf" config/includes.chroot/etc/sysctl.d/

mkdir -p config/includes.chroot/etc/zernel

# ============================================================
# Step 6: Add systemd services
# ============================================================
echo "[6/7] Adding systemd services..."
mkdir -p config/includes.chroot/lib/systemd/system
cp "$REPO_ROOT/distro/systemd/zerneld.service" config/includes.chroot/lib/systemd/system/
cp "$REPO_ROOT/distro/systemd/zernel-scheduler.service" config/includes.chroot/lib/systemd/system/
cp "$REPO_ROOT/distro/systemd/zernel-dashboard.service" config/includes.chroot/lib/systemd/system/
cp "$REPO_ROOT/distro/systemd/ollama.service" config/includes.chroot/lib/systemd/system/

# GNOME extension and branding (desktop profile only)
if [ "$PROFILE" = "desktop" ]; then
    echo "  Adding GNOME extension and branding..."
    mkdir -p config/includes.chroot/tmp/zernel-gnome
    cp -r "$REPO_ROOT/distro/gnome/zernel-gpu-indicator@dyber.io" config/includes.chroot/tmp/zernel-gnome/
    cp -r "$REPO_ROOT/distro/gnome/overrides" config/includes.chroot/tmp/zernel-gnome/
    cp "$REPO_ROOT/distro/iso/hooks/03-gnome-setup.hook.chroot" config/hooks/live/
    chmod +x config/hooks/live/03-gnome-setup.hook.chroot
fi

# Enable services on boot
cat > config/hooks/live/02-zernel-services.hook.chroot << 'HOOK'
#!/bin/bash
systemctl enable zerneld.service || true
systemctl enable zernel-scheduler.service || true
HOOK
chmod +x config/hooks/live/02-zernel-services.hook.chroot

# ============================================================
# Step 7: Build the ISO
# ============================================================
echo "[7/7] Building ISO (this may take 15-30 minutes)..."
lb build

if [ -f "live-image-${ARCH}.hybrid.iso" ]; then
    mv "live-image-${ARCH}.hybrid.iso" "$SCRIPT_DIR/$ISO_NAME"
    echo ""
    echo "ISO built successfully: $SCRIPT_DIR/$ISO_NAME"
    echo "Size: $(du -h "$SCRIPT_DIR/$ISO_NAME" | cut -f1)"
    echo ""
    echo "Write to USB: sudo dd if=$ISO_NAME of=/dev/sdX bs=4M status=progress"
else
    echo "ERROR: ISO build failed. Check logs above."
    exit 1
fi
