#!/usr/bin/env bash
# Zernel Installer
# Copyright (C) 2026 Dyber, Inc.
#
# Interactive installer for Zernel Linux OS.
# Run from the live ISO environment.
#
# Usage: zernel-install [--config unattended.yaml]
set -euo pipefail

VERSION="0.1.0"

echo "╔═══════════════════════════════════════╗"
echo "║     Zernel Installer v${VERSION}            ║"
echo "║     AI-Native Linux OS                ║"
echo "╚═══════════════════════════════════════╝"
echo ""

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Installer must be run as root."
    echo "Usage: sudo zernel-install"
    exit 1
fi

# ============================================================
# Step 1: Hardware Detection
# ============================================================
echo "[1/10] Detecting hardware..."
echo "  CPU: $(lscpu | grep 'Model name' | sed 's/.*:\s*//')"
echo "  RAM: $(free -h | awk '/Mem:/{print $2}')"
echo "  Cores: $(nproc)"

# Detect GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  GPUs: ${GPU_COUNT}x ${GPU_NAME}"
else
    echo "  GPUs: none detected (NVIDIA driver not loaded)"
    GPU_COUNT=0
fi

# Detect NUMA topology
NUMA_NODES=$(ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l)
echo "  NUMA nodes: ${NUMA_NODES}"

# Detect network
echo "  Network: $(ip -o link show | grep -v 'lo:' | awk '{print $2}' | tr '\n' ' ')"
echo ""

# ============================================================
# Step 2: Disk Selection
# ============================================================
echo "[2/10] Available disks:"
lsblk -d -o NAME,SIZE,TYPE,MODEL | grep disk
echo ""

read -rp "Install to disk (e.g., sda, nvme0n1): " TARGET_DISK
TARGET="/dev/${TARGET_DISK}"

if [ ! -b "$TARGET" ]; then
    echo "ERROR: ${TARGET} is not a block device."
    exit 1
fi

echo ""
echo "WARNING: ALL DATA on ${TARGET} will be erased!"
read -rp "Type 'yes' to continue: " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Installation cancelled."
    exit 0
fi

# ============================================================
# Step 3: Partitioning
# ============================================================
echo ""
echo "[3/10] Partitioning ${TARGET}..."
parted -s "$TARGET" mklabel gpt
parted -s "$TARGET" mkpart ESP fat32 1MiB 512MiB
parted -s "$TARGET" set 1 esp on
parted -s "$TARGET" mkpart primary ext4 512MiB 100%

# Detect partition naming (sda1 vs nvme0n1p1)
if [[ "$TARGET" == *"nvme"* ]]; then
    PART1="${TARGET}p1"
    PART2="${TARGET}p2"
else
    PART1="${TARGET}1"
    PART2="${TARGET}2"
fi

echo "[3/10] Formatting..."
mkfs.fat -F32 "$PART1"
mkfs.ext4 -F "$PART2"

# ============================================================
# Step 4: Mount and debootstrap
# ============================================================
echo "[4/10] Installing base system..."
MOUNT="/mnt/zernel"
mkdir -p "$MOUNT"
mount "$PART2" "$MOUNT"
mkdir -p "$MOUNT/boot/efi"
mount "$PART1" "$MOUNT/boot/efi"

debootstrap --arch=amd64 bookworm "$MOUNT" http://deb.debian.org/debian

# ============================================================
# Step 5: Configure chroot
# ============================================================
echo "[5/10] Configuring system..."
mount --bind /dev "$MOUNT/dev"
mount --bind /proc "$MOUNT/proc"
mount --bind /sys "$MOUNT/sys"

# Set hostname
echo "zernel" > "$MOUNT/etc/hostname"

# fstab
PART2_UUID=$(blkid -s UUID -o value "$PART2")
PART1_UUID=$(blkid -s UUID -o value "$PART1")
cat > "$MOUNT/etc/fstab" << EOF
UUID=${PART2_UUID}  /          ext4  errors=remount-ro  0  1
UUID=${PART1_UUID}  /boot/efi  vfat  umask=0077         0  1
EOF

# ============================================================
# Step 6: Install kernel and NVIDIA
# ============================================================
echo "[6/10] Installing kernel..."
chroot "$MOUNT" apt-get update -qq
chroot "$MOUNT" apt-get install -y --no-install-recommends \
    linux-image-amd64 firmware-linux grub-efi-amd64 \
    systemd-sysv locales openssh-server curl git

# ============================================================
# Step 7: Install Zernel binaries
# ============================================================
echo "[7/10] Installing Zernel..."
cp /usr/local/bin/zernel "$MOUNT/usr/local/bin/" 2>/dev/null || true
cp /usr/local/bin/zerneld "$MOUNT/usr/local/bin/" 2>/dev/null || true
cp /usr/local/bin/zernel-scheduler "$MOUNT/usr/local/bin/" 2>/dev/null || true

# ============================================================
# Step 8: Apply sysctl tuning
# ============================================================
echo "[8/10] Applying ML kernel tuning..."
mkdir -p "$MOUNT/etc/sysctl.d"
cp /etc/sysctl.d/99-zernel.conf "$MOUNT/etc/sysctl.d/" 2>/dev/null || true

mkdir -p "$MOUNT/etc/zernel"

# Systemd services
cp /lib/systemd/system/zerneld.service "$MOUNT/lib/systemd/system/" 2>/dev/null || true
cp /lib/systemd/system/zernel-scheduler.service "$MOUNT/lib/systemd/system/" 2>/dev/null || true
chroot "$MOUNT" systemctl enable zerneld.service 2>/dev/null || true
chroot "$MOUNT" systemctl enable zernel-scheduler.service 2>/dev/null || true

# ============================================================
# Step 9: Install bootloader
# ============================================================
echo "[9/10] Installing bootloader..."
chroot "$MOUNT" grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=zernel
chroot "$MOUNT" update-grub

# ============================================================
# Step 10: Create user and finalize
# ============================================================
echo "[10/10] Creating user account..."
read -rp "Username: " USERNAME
chroot "$MOUNT" useradd -m -s /bin/bash -G sudo "$USERNAME"
echo "Set password for ${USERNAME}:"
chroot "$MOUNT" passwd "$USERNAME"

# Cleanup
umount "$MOUNT/sys"
umount "$MOUNT/proc"
umount "$MOUNT/dev"
umount "$MOUNT/boot/efi"
umount "$MOUNT"

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  Installation complete!               ║"
echo "║  Remove installation media and reboot.║"
echo "╚═══════════════════════════════════════╝"
echo ""
echo "After reboot:"
echo "  zernel doctor    — verify your environment"
echo "  zernel watch     — GPU monitoring dashboard"
echo "  zernel run       — start training with tracking"
