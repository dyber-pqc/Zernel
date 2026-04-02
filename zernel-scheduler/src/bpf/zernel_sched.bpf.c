/* Copyright (C) 2026 Dyber, Inc. — SPDX-License-Identifier: GPL-2.0 */
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
extern void scx_bpf_dispatch(struct task_struct *p, u64 dsq_id,
                             u64 slice, u64 enq_flags) __ksym;
extern bool scx_bpf_consume(u64 dsq_id) __ksym;

/*
 * Built-in DSQ IDs from the kernel (include/linux/sched/ext.h)
 * SCX_DSQ_GLOBAL = (1ULL << 63) | 1
 * SCX_DSQ_LOCAL  = (1ULL << 63) | 2
 * We use our own user DSQ to support vtime dispatch in the future.
 */
#define ZERNEL_DSQ  0x5A45524E  /* "ZERN" */

/* ── Phase-aware scheduling ──────────────────────────────── */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u32);          /* pid */
    __type(value, u32);        /* phase */
    __uint(max_entries, 4096);
} phase_map SEC(".maps");

/* Per-CPU stats: [0]=enqueue, [1]=dispatch */
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

/* ── struct_ops callbacks ─────────────────────────────────── */

void BPF_STRUCT_OPS(zernel_enqueue, struct task_struct *p, u64 enq_flags)
{
    u32 pid = p->pid;
    u32 *phase = bpf_map_lookup_elem(&phase_map, &pid);
    u64 slice = phase ? phase_to_slice(*phase) : 5000000ULL;

    scx_bpf_dispatch(p, ZERNEL_DSQ, slice, enq_flags);
    stat_inc(0);
}

void BPF_STRUCT_OPS(zernel_dispatch, s32 cpu, struct task_struct *prev)
{
    scx_bpf_consume(ZERNEL_DSQ);
    stat_inc(1);
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
    /* Create our custom DSQ — NUMA node -1 = any */
    return scx_bpf_create_dsq(ZERNEL_DSQ, -1);
}

void BPF_STRUCT_OPS(zernel_exit, struct scx_exit_info *ei)
{
}

/* ── struct_ops registration ─────────────────────────────── */
SEC(".struct_ops.link")
struct sched_ext_ops zernel_ops = {
    .enqueue     = (void *)zernel_enqueue,
    .dispatch    = (void *)zernel_dispatch,
    .running     = (void *)zernel_running,
    .stopping    = (void *)zernel_stopping,
    .init        = (void *)zernel_init,
    .exit        = (void *)zernel_exit,
    .name        = "zernel",
};
