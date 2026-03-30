# Upgrade Guide

> Copyright (C) 2026 Dyber, Inc.

## Version Compatibility

| From | To | Breaking Changes | Migration Required |
|------|-----|-----------------|-------------------|
| 0.1.0-alpha | 0.1.0 | None | No |

## Upgrade Procedure

### Binary Upgrade (Manual Install)

```bash
# 1. Stop running services
sudo systemctl stop zernel-scheduler
sudo systemctl stop zerneld

# 2. Download new binaries
curl -fsSL https://github.com/dyber-pqc/Zernel/releases/latest/download/zernel-linux-x86_64.tar.gz | tar xz
cd zernel-linux-x86_64

# 3. Install
sudo cp zernel /usr/local/bin/
sudo cp zerneld /usr/local/bin/
sudo cp zernel-scheduler /usr/local/bin/

# 4. Verify
zernel --version
zerneld --version

# 5. Restart services
sudo systemctl start zerneld
sudo systemctl start zernel-scheduler

# 6. Check health
zernel doctor
curl localhost:9091/health
```

### Debian Package Upgrade

```bash
sudo apt update
sudo apt upgrade zernel-cli zerneld zernel-scheduler
```

### Docker Upgrade

```bash
docker pull ghcr.io/dyber-pqc/zernel:latest
docker compose up -d  # or restart your container
```

### Build from Source

```bash
git pull origin main
cargo build --workspace --release
# Then follow binary install steps above
```

## Data Migration

### Experiment Database

The SQLite experiment database at `~/.zernel/experiments/experiments.db` is forward-compatible. New columns are added with default values; old data is preserved.

**Schema versioning**: The database is automatically migrated on first access by each new version. No manual migration required.

If you need to back up:
```bash
cp ~/.zernel/experiments/experiments.db ~/.zernel/experiments/experiments.db.bak
```

### Model Registry

The model registry at `~/.zernel/models/registry.json` is a JSON file. New fields are added with defaults. Old entries work with new versions.

### Job Database

The jobs database at `~/.zernel/jobs/jobs.db` follows the same pattern as experiments.

### Configuration

`/etc/zernel/scheduler.toml` — new config fields use defaults if missing. Existing configs work without changes. To see new available options:

```bash
zernel-scheduler --dump-config
```

## Rollback

If an upgrade causes issues:

```bash
# 1. Stop services
sudo systemctl stop zernel-scheduler
sudo systemctl stop zerneld

# 2. Restore previous binaries (if you kept them)
sudo cp /usr/local/bin/zernel.bak /usr/local/bin/zernel
sudo cp /usr/local/bin/zerneld.bak /usr/local/bin/zerneld
sudo cp /usr/local/bin/zernel-scheduler.bak /usr/local/bin/zernel-scheduler

# 3. Restart
sudo systemctl start zerneld
sudo systemctl start zernel-scheduler
```

**Recommendation**: Before upgrading, back up your binaries:
```bash
for bin in zernel zerneld zernel-scheduler; do
    sudo cp /usr/local/bin/$bin /usr/local/bin/$bin.bak
done
```

## Kernel Upgrade

If upgrading the Linux kernel (e.g., from 6.12 to 6.13):

1. Install the new kernel package
2. Verify `CONFIG_SCHED_CLASS_EXT=y` is set
3. Reboot
4. The Zernel scheduler will automatically reload on boot via systemd
5. Run `zernel doctor` to verify

The Zernel sysctl parameters (`/etc/sysctl.d/99-zernel.conf`) are applied automatically on each boot and do not need updating for kernel upgrades.
