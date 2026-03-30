# Kernel Configuration

> Copyright (C) 2026 Dyber, Inc.

## Overview

Zernel ships a heavily customized Linux kernel `.config` that bakes in ML-optimized parameters. Users never need to configure this manually -- it ships as part of the Zernel distro.

## Base Distribution

Zernel builds on **Debian stable** (not Ubuntu):
- Longer release cycle -- more stable for kernel customization
- No Canonical dependencies
- Strong server/enterprise lineage
- `debootstrap` makes custom ISO builds straightforward

## Kernel Version

- **Minimum**: Linux 6.12 (first stable release with `sched_ext`)
- **Target**: Latest stable kernel in the 6.x series

## Key Kernel Configuration

### sched_ext Framework

```
CONFIG_SCHED_CLASS_EXT=y
CONFIG_BPF=y
CONFIG_BPF_SYSCALL=y
CONFIG_BPF_JIT=y
CONFIG_BPF_JIT_ALWAYS_ON=y
```

### Memory Management

```
# Transparent huge pages for ML memory allocation patterns
CONFIG_TRANSPARENT_HUGEPAGE=y
CONFIG_TRANSPARENT_HUGEPAGE_ALWAYS=y

# 1GB huge pages for GPU DMA buffers
CONFIG_HUGETLB_PAGE=y
CONFIG_HUGETLBFS=y

# High-resolution timers for accurate profiling
CONFIG_HZ_1000=y
CONFIG_HZ=1000
CONFIG_HIGH_RES_TIMERS=y
```

### NUMA Support

```
CONFIG_NUMA=y
CONFIG_NUMA_BALANCING=y
CONFIG_NUMA_BALANCING_DEFAULT_ENABLED=y
CONFIG_MEMPOLICY=y
```

### Network Stack (Distributed Training)

```
# RDMA / InfiniBand
CONFIG_INFINIBAND=y
CONFIG_INFINIBAND_USER_ACCESS=y
CONFIG_RDMA_RXE=y

# BBR congestion control
CONFIG_TCP_CONG_BBR=y
```

### I/O for Dataset Loading

```
CONFIG_IO_URING=y
CONFIG_BLK_DEV_NVME=y
CONFIG_NVME_MULTIPATH=y
```

### Container Support

```
CONFIG_NAMESPACES=y
CONFIG_CGROUPS=y
CONFIG_CGROUP_SCHED=y
CONFIG_OVERLAY_FS=y
```

## sysctl Tuning

Applied at boot via `/etc/sysctl.d/99-zernel.conf`:

```bash
# Virtual Memory
vm.swappiness=0                    # Never swap
vm.dirty_ratio=40                  # Allow more dirty pages for large dataset writes
vm.dirty_background_ratio=10
vm.overcommit_memory=1             # Allow overcommit (PyTorch needs this)
vm.nr_hugepages=1024               # Pre-allocate 1GB huge pages

# Network (Distributed Training)
net.core.rmem_max=134217728        # 128MB receive buffer
net.core.wmem_max=134217728        # 128MB send buffer
net.ipv4.tcp_rmem=4096 87380 134217728
net.ipv4.tcp_wmem=4096 65536 134217728
net.core.netdev_max_backlog=250000
net.ipv4.tcp_congestion_control=bbr

# NUMA
kernel.numa_balancing=1

# File Handles
fs.file-max=2097152
fs.inotify.max_user_watches=524288
```

## NVIDIA Stack

Pre-installed in the Zernel ISO:

| Package | Purpose |
|---------|---------|
| nvidia-open-kernel-modules | Open-source GPU kernel modules |
| cuda-toolkit-12-6 | CUDA Toolkit |
| libcudnn9 | Deep neural network acceleration |
| tensorrt | Inference optimization |
| libnccl2 | Multi-GPU collective communications |
| nvidia-container-toolkit | Docker/containerd GPU support |
| nvidia-fabricmanager | NVSwitch systems |

## Pre-installed ML Frameworks

```
python3.12
torch (latest stable, CUDA build)
jax + jaxlib (CUDA build)
transformers (HuggingFace)
vllm
accelerate
deepspeed
```

All framework versions are tested together on the Zernel kernel config before each release.
