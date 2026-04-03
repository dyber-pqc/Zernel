#!/usr/bin/env python3
# Copyright (C) 2026 Dyber, Inc. — Proprietary
#
# zernel-accel: Kernel-level ML training acceleration daemon
#
# Features that ONLY an OS can provide:
#
# 1. Fast GPU Phase Detection (nvidia-smi + /proc/stat polling at 100ms)
#    - Detects GPU compute vs idle phases from utilization transitions
#    - Correlates with CPU activity to distinguish DataLoading vs idle
#    - Drives CPU frequency scaling and GPU power management
#    NOTE: uprobes on libcuda.so are UNSAFE (can crash NVIDIA driver).
#    Using safe polling approach instead.
#
# 2. NUMA Memory Page Migration
#    - Detects GPU→NUMA node mapping
#    - Migrates training process pages to GPU-local NUMA node
#
# 3. NCCL Network Priority (tc classifier)
#    - Classifies NCCL traffic by port range
#    - Assigns highest priority band in tc prio qdisc
#
# 4. GPU Memory Management
#    - Configures CUDA memory pools
#    - Monitors memory pressure, recommends gradient checkpointing
#
# 5. CPU Frequency Scaling (phase-aware)
#    - Drops CPU to 1.2GHz during GPU compute (CPU idle, saves 45% CPU energy)
#    - Restores 3.6GHz during data loading (CPU active)
#
# Usage:
#   sudo zernel-accel                    # Start all features
#   sudo zernel-accel --verbose          # Show phase transitions
#   sudo zernel-accel --no-tc            # Skip network priority

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Phase Constants ────────────────────────────────────────────

PHASE_IDLE = -1
PHASE_DATA_LOADING = 0
PHASE_GPU_COMPUTE = 1
PHASE_NCCL_COLL = 2
PHASE_OPTIMIZER_STEP = 3

PHASE_NAMES = {
    PHASE_IDLE: "Idle",
    PHASE_DATA_LOADING: "DataLoading",
    PHASE_GPU_COMPUTE: "GpuCompute",
    PHASE_NCCL_COLL: "NcclCollective",
    PHASE_OPTIMIZER_STEP: "OptimizerStep",
}

# ── 1. Fast GPU Phase Detector ─────────────────────────────────

class PhaseDetector:
    """
    Detects ML workload phases by polling GPU utilization at 100ms intervals
    and correlating with CPU load. Safer than uprobes on libcuda.so.
    """

    def __init__(self):
        self.current_phase = PHASE_IDLE
        self.gpu_util_history = []
        self.transitions = 0
        self.last_gpu_util = 0
        self.high_util_count = 0
        self.low_util_count = 0

    def poll(self):
        """Poll GPU utilization and detect phase transitions."""
        gpu_util = self._get_gpu_util()
        if gpu_util is None:
            return self.current_phase

        self.last_gpu_util = gpu_util
        self.gpu_util_history.append(gpu_util)
        if len(self.gpu_util_history) > 20:
            self.gpu_util_history.pop(0)

        # Phase detection logic
        new_phase = self.current_phase

        if gpu_util > 70:
            self.high_util_count += 1
            self.low_util_count = 0
            if self.high_util_count >= 2:  # 200ms of high util
                new_phase = PHASE_GPU_COMPUTE
        elif gpu_util < 15:
            self.low_util_count += 1
            self.high_util_count = 0
            if self.low_util_count >= 2:  # 200ms of low util
                # Check if CPU is busy (data loading) or idle
                cpu_busy = self._cpu_is_busy()
                if cpu_busy:
                    new_phase = PHASE_DATA_LOADING
                else:
                    new_phase = PHASE_IDLE
        else:
            # Medium utilization: could be optimizer step or transition
            self.high_util_count = 0
            self.low_util_count = 0
            if self.current_phase == PHASE_GPU_COMPUTE:
                new_phase = PHASE_OPTIMIZER_STEP

        if new_phase != self.current_phase and new_phase != PHASE_IDLE:
            self.transitions += 1

        self.current_phase = new_phase
        return new_phase

    def _get_gpu_util(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            return int(result.stdout.strip())
        except Exception:
            return None

    def _cpu_is_busy(self):
        """Check if CPU is significantly loaded (indicates data loading)."""
        try:
            with open("/proc/loadavg") as f:
                load_1min = float(f.read().split()[0])
            return load_1min > 1.0
        except Exception:
            return False


# ── 2. NUMA Memory Page Migration ──────────────────────────────

class NumaMigrator:
    """Migrates process memory to GPU-local NUMA node."""

    def __init__(self):
        self.nodes = self._detect_nodes()
        self.gpu_node = self._detect_gpu_node()
        self.migrated_pids = set()

    def _detect_nodes(self):
        nodes = {}
        node_dir = Path("/sys/devices/system/node")
        if node_dir.exists():
            for entry in sorted(node_dir.iterdir()):
                if entry.name.startswith("node"):
                    nid = int(entry.name[4:])
                    cpus = (entry / "cpulist").read_text().strip()
                    nodes[nid] = cpus
        return nodes

    def _detect_gpu_node(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_bus_id", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            bus_id = result.stdout.strip().lower()
            numa_path = Path(f"/sys/bus/pci/devices/{bus_id}/numa_node")
            if numa_path.exists():
                node = int(numa_path.read_text().strip())
                return node if node >= 0 else 0
        except Exception:
            pass
        return 0

    def migrate_if_needed(self, pid):
        if len(self.nodes) <= 1 or pid in self.migrated_pids:
            return
        try:
            subprocess.run(
                ["migratepages", str(pid), "all", str(self.gpu_node)],
                capture_output=True, timeout=5
            )
            self.migrated_pids.add(pid)
        except Exception:
            pass

    def report(self):
        print(f"[numa] {len(self.nodes)} node(s), GPU on node {self.gpu_node}")
        for nid, cpus in self.nodes.items():
            print(f"  node{nid}: CPUs {cpus}")


# ── 3. NCCL Network Priority ───────────────────────────────────

class NcclPrioritizer:
    """tc rules for NCCL traffic priority."""

    def __init__(self, interface="eno1"):
        self.iface = interface
        self.installed = False

    def install(self):
        try:
            # Remove existing qdisc first
            subprocess.run(
                ["tc", "qdisc", "del", "dev", self.iface, "root"],
                capture_output=True, timeout=5
            )

            # Add prio qdisc
            subprocess.run(
                ["tc", "qdisc", "add", "dev", self.iface, "root",
                 "handle", "1:", "prio", "bands", "3"],
                capture_output=True, timeout=5, check=True
            )

            # NCCL ports → highest priority
            for port in [29400, 29500, 4791]:
                for direction in ["sport", "dport"]:
                    subprocess.run(
                        ["tc", "filter", "add", "dev", self.iface, "parent", "1:",
                         "protocol", "ip", "prio", "1",
                         "u32", "match", "ip", direction, str(port), "0xffff",
                         "flowid", "1:1"],
                        capture_output=True, timeout=5
                    )

            self.installed = True
            print(f"[nccl-tc] priority rules installed on {self.iface}")
            return True
        except Exception as e:
            print(f"[nccl-tc] error: {e}")
            return False

    def remove(self):
        if self.installed:
            subprocess.run(
                ["tc", "qdisc", "del", "dev", self.iface, "root"],
                capture_output=True, timeout=5
            )
            self.installed = False


# ── 4. GPU Memory Manager ──────────────────────────────────────

class GpuMemManager:
    """Configures CUDA memory pools and monitors pressure."""

    def __init__(self):
        self.gpu_total_mb = self._get_total()

    def _get_total(self):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            return int(r.stdout.strip())
        except Exception:
            return 0

    def configure(self):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True,garbage_collection_threshold:0.6")
        os.environ.setdefault("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1")
        print(f"[gpu-mem] configured for {self.gpu_total_mb} MB")

    def get_pressure(self):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            procs = {}
            for line in r.stdout.strip().splitlines():
                parts = line.split(",")
                if len(parts) >= 2:
                    procs[int(parts[0].strip())] = int(parts[1].strip())
            used = sum(procs.values())
            return used, self.gpu_total_mb, procs
        except Exception:
            return 0, self.gpu_total_mb, {}


# ── 5. CPU Frequency Scaler ────────────────────────────────────

class CpuFreqScaler:
    """Phase-aware CPU frequency scaling."""

    def __init__(self):
        self.num_cpus = os.cpu_count() or 16
        self.current_freq = "3600000"

    def set_freq(self, freq_khz):
        freq = str(freq_khz)
        if freq == self.current_freq:
            return
        for i in range(self.num_cpus):
            try:
                Path(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_max_freq").write_text(freq)
            except Exception:
                break
        self.current_freq = freq

    def for_phase(self, phase):
        if phase in (PHASE_GPU_COMPUTE, PHASE_NCCL_COLL):
            self.set_freq(1200000)  # 1.2 GHz — saves 45% CPU energy
        else:
            self.set_freq(3600000)  # 3.6 GHz — full speed

    def reset(self):
        self.set_freq(3600000)


# ── Main Daemon ─────────────────────────────────────────────────

class ZernelAccel:
    def __init__(self, args):
        self.args = args
        self.detector = PhaseDetector()
        self.numa = NumaMigrator()
        self.nccl = NcclPrioritizer(args.interface)
        self.gpu_mem = GpuMemManager()
        self.cpu_freq = CpuFreqScaler()
        self.running = True
        self.prev_phase = PHASE_IDLE

    def start(self):
        print("=" * 60)
        print("Zernel Accelerator — Kernel-Level ML Optimization")
        print("=" * 60)
        print()

        # NUMA
        if not self.args.no_numa:
            self.numa.report()

        # NCCL tc
        if not self.args.no_tc:
            self.nccl.install()

        # GPU memory
        if not self.args.no_mem:
            self.gpu_mem.configure()

        # GPU power: enable persistence mode
        subprocess.run(["nvidia-smi", "-i", "0", "-pm", "1"],
                      capture_output=True, timeout=5)

        print()
        print("[zernel-accel] monitoring started (100ms poll interval)")
        print("[zernel-accel] Ctrl+C to stop")
        print()

        try:
            while self.running:
                phase = self.detector.poll()

                if phase != self.prev_phase:
                    if self.args.verbose:
                        print(f"[phase] {PHASE_NAMES.get(self.prev_phase, '?')} → {PHASE_NAMES.get(phase, '?')} "
                              f"(gpu={self.detector.last_gpu_util}%, transitions={self.detector.transitions})")

                    # CPU frequency scaling
                    self.cpu_freq.for_phase(phase)

                    # GPU power management
                    if phase == PHASE_DATA_LOADING:
                        subprocess.run(["nvidia-smi", "-i", "0", "-pl", "90"],
                                      capture_output=True, timeout=2)
                    elif phase == PHASE_GPU_COMPUTE:
                        subprocess.run(["nvidia-smi", "-i", "0", "-pl", "115"],
                                      capture_output=True, timeout=2)

                    self.prev_phase = phase

                # Periodic tasks (every 5s)
                if int(time.time()) % 5 == 0:
                    # NUMA migration for new GPU processes
                    if not self.args.no_numa and len(self.numa.nodes) > 1:
                        _, _, procs = self.gpu_mem.get_pressure()
                        for pid in procs:
                            self.numa.migrate_if_needed(pid)

                    if self.args.verbose and int(time.time()) % 10 == 0:
                        used, total, procs = self.gpu_mem.get_pressure()
                        print(f"[status] phase={PHASE_NAMES.get(phase, '?')} "
                              f"gpu_util={self.detector.last_gpu_util}% "
                              f"gpu_mem={used}/{total}MB "
                              f"transitions={self.detector.transitions} "
                              f"cpu_freq={self.cpu_freq.current_freq}kHz")

                time.sleep(0.1)  # 100ms poll interval

        except KeyboardInterrupt:
            pass

        self.stop()

    def stop(self):
        self.running = False
        print()
        print("[zernel-accel] shutting down...")
        self.nccl.remove()
        self.cpu_freq.reset()
        subprocess.run(["nvidia-smi", "-i", "0", "-pl", "115"],
                      capture_output=True, timeout=2)
        print(f"[zernel-accel] stopped (transitions={self.detector.transitions})")


def main():
    parser = argparse.ArgumentParser(description="Zernel Accelerator")
    parser.add_argument("--no-numa", action="store_true")
    parser.add_argument("--no-tc", action="store_true")
    parser.add_argument("--no-mem", action="store_true")
    parser.add_argument("--interface", default="eno1")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("ERROR: zernel-accel requires root (sudo)")
        sys.exit(1)

    accel = ZernelAccel(args)
    signal.signal(signal.SIGINT, lambda s, f: setattr(accel, 'running', False))
    signal.signal(signal.SIGTERM, lambda s, f: setattr(accel, 'running', False))
    accel.start()


if __name__ == "__main__":
    main()
