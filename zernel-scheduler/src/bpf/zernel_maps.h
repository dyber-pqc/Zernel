/* Copyright (C) 2026 Dyber, Inc. — GPL-2.0 */
/* Shared BPF map definitions for Zernel scheduler */

#ifndef __ZERNEL_MAPS_H
#define __ZERNEL_MAPS_H

/* Workload phase enum — must match Rust WorkloadPhase */
enum zernel_phase {
    PHASE_DATA_LOADING   = 0,
    PHASE_GPU_COMPUTE    = 1,
    PHASE_NCCL_COLL      = 2,
    PHASE_OPTIMIZER_STEP = 3,
    PHASE_UNKNOWN        = 4,
};

/* Per-task state stored in BPF map */
struct zernel_task_state {
    __u32 pid;
    __u8  is_ml_process;
    __u8  current_phase;     /* enum zernel_phase */
    __u8  gpu_utilization;   /* 0-100 */
    __u8  _pad;
    __u64 last_gpu_sync_ns;
    __u64 cpu_burst_duration_ns;
    __u32 io_wait_fraction;  /* fixed-point: value * 1000 */
};

/* Scheduling decision written by userspace, read by BPF */
struct zernel_sched_decision {
    __s32 priority;
    __u8  preempt;
    __u8  _pad[3];
    __u64 latency_target_ns;
};

#endif /* __ZERNEL_MAPS_H */
