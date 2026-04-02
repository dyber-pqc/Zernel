/* Copyright (C) 2026 Dyber, Inc. — SPDX-License-Identifier: GPL-2.0 */
/*
 * Zernel ML-Aware sched_ext Scheduler (v2)
 *
 * Architecture:
 *   - select_cpu: prefer idle CPUs, dispatch directly to SCX_DSQ_LOCAL
 *     to preserve cache locality (the #1 performance win over v1)
 *   - enqueue: only tasks that miss select_cpu go to the shared DSQ,
 *     with phase-aware time slices for ML workloads
 *   - dispatch: consume from the shared DSQ as fallback
 *   - running/stopping: emit lifecycle events to ring buffer for
 *     userspace phase detection
 *
 * Phase-aware time slices:
 *   0 = DataLoading    — 5 ms  (default, many small I/O bursts)
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

/*
 * Built-in DSQ constants from include/linux/sched/ext.h:
 *   SCX_DSQ_LOCAL  = (1ULL << 63) | 2
 *   SCX_DSQ_GLOBAL = (1ULL << 63) | 1
 */
#define SCX_DSQ_LOCAL_VAL  ((1ULL << 63) | 2)
#define SCX_DSQ_GLOBAL_VAL ((1ULL << 63) | 1)

/* Our shared fallback DSQ (for tasks that miss select_cpu) */
#define ZERNEL_DSQ  0x5A45524E  /* "ZERN" */

/* Default slice when no phase info is available */
#define SCX_SLICE_DFL  20000000ULL   /* 20 ms */

/* ── Phase-aware scheduling ──────────────────────────────── */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u32);          /* pid */
    __type(value, u32);        /* phase */
    __uint(max_entries, 4096);
} phase_map SEC(".maps");

/* Per-CPU stats: [0]=local_dispatch, [1]=global_dispatch, [2]=consume */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(u64));
    __uint(max_entries, 4);
} stats SEC(".maps");

/* Ring buffer for task lifecycle events */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} task_events SEC(".maps");

struct zernel_task_event {
    __u32 pid;
    __u8  event_type;   /* 0=running, 1=stopping */
    __u8  phase;
    __u8  _pad[2];
    __u64 timestamp_ns;
};

static void stat_inc(u32 idx)
{
    u64 *cnt = bpf_map_lookup_elem(&stats, &idx);
    if (cnt)
        __sync_fetch_and_add(cnt, 1);
}

static inline u64 phase_to_slice(u32 phase)
{
    switch (phase) {
    case 1:  return 20000000ULL;  /* GpuCompute    — 20 ms */
    case 2:  return 10000000ULL;  /* NcclCollective — 10 ms */
    case 3:  return  3000000ULL;  /* OptimizerStep  — 3 ms */
    default: return  5000000ULL;  /* DataLoading    — 5 ms */
    }
}

static inline u64 get_task_slice(struct task_struct *p)
{
    u32 pid = p->pid;
    u32 *phase = bpf_map_lookup_elem(&phase_map, &pid);
    return phase ? phase_to_slice(*phase) : SCX_SLICE_DFL;
}

/* ── struct_ops callbacks ─────────────────────────────────── */

/*
 * select_cpu: Called when a task is waking up. Try to find an idle CPU
 * (preferring the previous CPU for cache locality) and dispatch directly
 * to SCX_DSQ_LOCAL, bypassing the global queue entirely.
 *
 * This is THE critical optimization — it keeps hot caches warm and avoids
 * global queue contention.
 */
s32 BPF_STRUCT_OPS(zernel_select_cpu, struct task_struct *p, s32 prev_cpu,
                   u64 wake_flags)
{
    bool is_idle = false;
    s32 cpu;

    cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);
    if (is_idle) {
        u64 slice = get_task_slice(p);
        scx_bpf_dispatch(p, SCX_DSQ_LOCAL_VAL, slice, 0);
        stat_inc(0);  /* local dispatch */
    }

    return cpu;
}

/*
 * enqueue: Called for tasks that were NOT dispatched in select_cpu
 * (i.e. no idle CPU was found). Send them to our shared DSQ with
 * a phase-aware time slice.
 */
void BPF_STRUCT_OPS(zernel_enqueue, struct task_struct *p, u64 enq_flags)
{
    u64 slice = get_task_slice(p);
    scx_bpf_dispatch(p, ZERNEL_DSQ, slice, enq_flags);
    stat_inc(1);  /* global dispatch */
}

/*
 * dispatch: Called when a CPU has no work. Pull from our shared DSQ.
 */
void BPF_STRUCT_OPS(zernel_dispatch, s32 cpu, struct task_struct *prev)
{
    scx_bpf_consume(ZERNEL_DSQ);
    stat_inc(2);  /* consume */
}

void BPF_STRUCT_OPS(zernel_running, struct task_struct *p)
{
    struct zernel_task_event *e;
    e = bpf_ringbuf_reserve(&task_events, sizeof(*e), 0);
    if (!e)
        return;
    e->pid = p->pid;
    e->event_type = 0;
    e->phase = 0;
    u32 pid = p->pid;
    u32 *ph = bpf_map_lookup_elem(&phase_map, &pid);
    if (ph)
        e->phase = (u8)*ph;
    e->timestamp_ns = bpf_ktime_get_ns();
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
    e->phase = 0;
    u32 pid = p->pid;
    u32 *ph = bpf_map_lookup_elem(&phase_map, &pid);
    if (ph)
        e->phase = (u8)*ph;
    e->timestamp_ns = bpf_ktime_get_ns();
    bpf_ringbuf_submit(e, 0);
}

s32 BPF_STRUCT_OPS_SLEEPABLE(zernel_init)
{
    return scx_bpf_create_dsq(ZERNEL_DSQ, -1);
}

void BPF_STRUCT_OPS(zernel_exit, struct scx_exit_info *ei)
{
}

/* ── struct_ops registration ─────────────────────────────── */
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

