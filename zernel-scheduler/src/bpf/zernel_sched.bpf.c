/* Copyright (C) 2026 Dyber, Inc. — SPDX-License-Identifier: GPL-2.0 */
/*
 * Zernel ML-Aware sched_ext Scheduler (v3)
 *
 * Architecture:
 *   - Per-CPU dispatch queues via select_cpu + SCX_DSQ_LOCAL
 *   - Shared fallback DSQ only when no idle CPU is available
 *   - Phase-aware time slices from phase_map (written by userspace)
 *   - Preemption control: GPU compute tasks get preempt flag cleared
 *   - Ring buffer events for userspace phase detection and GPU power mgmt
 *   - CPU affinity hints via cpu_affinity_map (written by userspace)
 *
 * Phase time slices:
 *   0 = DataLoading    — 5 ms  (many small I/O bursts, high preemption)
 *   1 = GpuCompute     — 20 ms (long slice, minimize preemption)
 *   2 = NcclCollective — 10 ms (network-sensitive, medium slice)
 *   3 = OptimizerStep  — 3 ms  (latency-sensitive CPU burst)
 */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char _license[] SEC("license") = "GPL";

/* Polyfill for libbpf < 1.3 */
#ifndef BPF_STRUCT_OPS
#define BPF_STRUCT_OPS(name, args...)      \
    SEC("struct_ops/"#name)                \
    BPF_PROG(name, ##args)
#endif
#ifndef BPF_STRUCT_OPS_SLEEPABLE
#define BPF_STRUCT_OPS_SLEEPABLE(name, args...) \
    SEC("struct_ops.s/"#name)                   \
    BPF_PROG(name, ##args)
#endif

/* ── kfunc externs ─────────────────────────────────────────── */
extern s32 scx_bpf_create_dsq(u64 dsq_id, s32 node) __ksym;
extern s32 scx_bpf_select_cpu_dfl(struct task_struct *p, s32 prev_cpu,
                                   u64 wake_flags, bool *is_idle) __ksym;
extern void scx_bpf_dispatch(struct task_struct *p, u64 dsq_id,
                             u64 slice, u64 enq_flags) __ksym;
extern bool scx_bpf_consume(u64 dsq_id) __ksym;

/* Built-in DSQ constants */
#define SCX_DSQ_LOCAL_VAL  ((1ULL << 63) | 2)

/* Our shared fallback DSQ */
#define ZERNEL_DSQ  0x5A45524E  /* "ZERN" */

/* Default slice */
#define SCX_SLICE_DFL  20000000ULL   /* 20 ms */

/* Phase constants */
#define PHASE_DATA_LOADING   0
#define PHASE_GPU_COMPUTE    1
#define PHASE_NCCL_COLL      2
#define PHASE_OPTIMIZER_STEP 3

/* ── BPF Maps ──────────────────────────────────────────────── */

/* Phase map: pid -> phase_id (written by userspace phase detector) */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u32);
    __type(value, u32);
    __uint(max_entries, 4096);
} phase_map SEC(".maps");

/*
 * CPU affinity map: pid -> preferred_cpu
 * Written by userspace for data-loading threads feeding a specific GPU.
 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u32);
    __type(value, s32);
    __uint(max_entries, 4096);
} cpu_affinity_map SEC(".maps");

/* Per-CPU stats: [0]=local, [1]=global, [2]=consume, [3]=preempt_blocked */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(u64));
    __uint(max_entries, 8);
} stats SEC(".maps");

/* Ring buffer for task lifecycle events (read by userspace) */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} task_events SEC(".maps");

struct zernel_task_event {
    __u32 pid;
    __u8  event_type;   /* 0=running, 1=stopping */
    __u8  phase;
    __u16 cpu;
    __u64 timestamp_ns;
    __u64 runtime_ns;
};

/* ── Helpers ───────────────────────────────────────────────── */

static void stat_inc(u32 idx)
{
    u64 *cnt = bpf_map_lookup_elem(&stats, &idx);
    if (cnt)
        __sync_fetch_and_add(cnt, 1);
}

static inline u64 phase_to_slice(u32 phase)
{
    switch (phase) {
    case PHASE_GPU_COMPUTE:    return 20000000ULL;  /* 20 ms */
    case PHASE_NCCL_COLL:      return 10000000ULL;  /* 10 ms */
    case PHASE_OPTIMIZER_STEP: return  3000000ULL;  /* 3 ms */
    case PHASE_DATA_LOADING:   return  5000000ULL;  /* 5 ms */
    default:                   return SCX_SLICE_DFL; /* 20 ms */
    }
}

static inline u64 get_task_slice(struct task_struct *p)
{
    u32 pid = p->pid;
    u32 *phase = bpf_map_lookup_elem(&phase_map, &pid);
    return phase ? phase_to_slice(*phase) : SCX_SLICE_DFL;
}

static inline u32 get_task_phase(struct task_struct *p)
{
    u32 pid = p->pid;
    u32 *phase = bpf_map_lookup_elem(&phase_map, &pid);
    return phase ? *phase : 255;
}

/* ── struct_ops callbacks ─────────────────────────────────── */

s32 BPF_STRUCT_OPS(zernel_select_cpu, struct task_struct *p, s32 prev_cpu,
                   u64 wake_flags)
{
    bool is_idle = false;
    s32 cpu;

    cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);

    if (is_idle) {
        u64 slice = get_task_slice(p);
        scx_bpf_dispatch(p, SCX_DSQ_LOCAL_VAL, slice, 0);
        stat_inc(0);
    }

    return cpu;
}

void BPF_STRUCT_OPS(zernel_enqueue, struct task_struct *p, u64 enq_flags)
{
    u64 slice = get_task_slice(p);
    u32 phase = get_task_phase(p);

    /* Preemption control: don't preempt GPU compute or NCCL tasks */
    if (phase == PHASE_GPU_COMPUTE || phase == PHASE_NCCL_COLL) {
        enq_flags &= ~0x10000000ULL;
        stat_inc(3);
    }

    scx_bpf_dispatch(p, ZERNEL_DSQ, slice, enq_flags);
    stat_inc(1);
}

void BPF_STRUCT_OPS(zernel_dispatch, s32 cpu, struct task_struct *prev)
{
    scx_bpf_consume(ZERNEL_DSQ);
    stat_inc(2);
}

void BPF_STRUCT_OPS(zernel_running, struct task_struct *p)
{
    struct zernel_task_event *e;
    e = bpf_ringbuf_reserve(&task_events, sizeof(*e), 0);
    if (!e)
        return;
    e->pid = p->pid;
    e->event_type = 0;
    e->phase = (u8)get_task_phase(p);
    e->cpu = bpf_get_smp_processor_id();
    e->timestamp_ns = bpf_ktime_get_ns();
    e->runtime_ns = 0;
    bpf_ringbuf_submit(e, 0);
}

void BPF_STRUCT_OPS(zernel_stopping, struct task_struct *p, bool runnable)
{
    struct zernel_task_event *e;
    e = bpf_ringbuf_reserve(&task_events, sizeof(*e), 0);
    if (!e)
        return;
    e->pid = p->pid;
    e->event_type = 1;
    e->phase = (u8)get_task_phase(p);
    e->cpu = bpf_get_smp_processor_id();
    e->timestamp_ns = bpf_ktime_get_ns();
    e->runtime_ns = p->se.sum_exec_runtime;
    bpf_ringbuf_submit(e, 0);
}

s32 BPF_STRUCT_OPS_SLEEPABLE(zernel_init)
{
    return scx_bpf_create_dsq(ZERNEL_DSQ, -1);
}

void BPF_STRUCT_OPS(zernel_exit, struct scx_exit_info *ei)
{
}

SEC(".struct_ops.link")
struct sched_ext_ops zernel_ops = {
    .select_cpu  = (void *)zernel_select_cpu,
    .enqueue     = (void *)zernel_enqueue,
    .dispatch    = (void *)zernel_dispatch,
    .running     = (void *)zernel_running,
    .stopping    = (void *)zernel_stopping,
    .init        = (void *)zernel_init,
    .exit        = (void *)zernel_exit,
    .name        = "zernel",
};
