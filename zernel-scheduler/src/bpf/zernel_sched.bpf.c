/* Copyright (C) 2026 Dyber, Inc. — GPL-2.0 */
/*
 * Zernel sched_ext BPF Scheduler
 *
 * Implements all 8 sched_ext ops for ML workload-aware CPU scheduling.
 * Reads scheduling decisions from BPF maps written by the userspace
 * Rust daemon (phase detector + NUMA + multi-tenant).
 *
 * Requires: Linux 6.12+ with CONFIG_SCHED_CLASS_EXT=y
 *
 * Maps:
 *   task_states     — per-PID task state (written by userspace)
 *   sched_decisions — per-PID scheduling decisions (written by userspace)
 *   task_events     — ring buffer for task lifecycle events (read by userspace)
 */

#ifdef __BPF__
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#endif

#include "zernel_maps.h"

/* ============================================================
 * BPF Maps
 * ============================================================ */

#ifdef __BPF__

/* Per-PID task state (written by userspace phase detector) */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, __u32);
    __type(value, struct zernel_task_state);
} task_states SEC(".maps");

/* Per-PID scheduling decisions (written by userspace scheduler) */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, __u32);
    __type(value, struct zernel_sched_decision);
} sched_decisions SEC(".maps");

/* Task lifecycle events (read by userspace for phase detection) */
struct zernel_task_event {
    __u32 pid;
    __u8  event_type;  /* 0 = running, 1 = stopping, 2 = new_task */
    __u8  _pad[3];
    __u64 timestamp_ns;
    __u64 runtime_ns;  /* for stopping events: how long the task ran */
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} task_events SEC(".maps");

/* Default time slice (5ms) */
#define DEFAULT_SLICE_NS  (5ULL * 1000 * 1000)

/* Boosted slice for high-priority phases (2ms — run more frequently) */
#define HIGH_PRIO_SLICE_NS (2ULL * 1000 * 1000)

/* Low-priority slice (20ms — yield CPU) */
#define LOW_PRIO_SLICE_NS (20ULL * 1000 * 1000)

/* ============================================================
 * sched_ext Operations
 * ============================================================ */

/*
 * select_cpu — Choose which CPU to run a task on.
 *
 * Reads the preferred_cpu from the sched_decisions map. If set and idle,
 * use it (NUMA-aware placement from userspace). Otherwise, fall back to
 * the default CPU selection.
 */
s32 BPF_STRUCT_OPS(zernel_select_cpu, struct task_struct *p,
                   s32 prev_cpu, u64 wake_flags)
{
    __u32 pid = p->pid;
    struct zernel_sched_decision *decision;

    decision = bpf_map_lookup_elem(&sched_decisions, &pid);
    if (decision && decision->latency_target_ns > 0) {
        /* Userspace set a preferred CPU (NUMA-aware) */
        s32 preferred = (__s32)(decision->latency_target_ns & 0xFFFF);
        /* Use latency_target_ns low bits as CPU hint
         * (actual preferred_cpu field would be added in production) */
        if (preferred >= 0 && scx_bpf_test_and_clear_cpu_idle(preferred))
            return preferred;
    }

    return scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, NULL);
}

/*
 * enqueue — Task becomes runnable; add it to a dispatch queue.
 *
 * Reads the scheduling decision. High-priority tasks get shorter slices
 * (scheduled more frequently) and may preempt. Low-priority tasks (GPU
 * compute phase) get long slices and yield.
 */
void BPF_STRUCT_OPS(zernel_enqueue, struct task_struct *p, u64 enq_flags)
{
    __u32 pid = p->pid;
    struct zernel_sched_decision *decision;
    __u64 slice = DEFAULT_SLICE_NS;
    u64 flags = 0;

    decision = bpf_map_lookup_elem(&sched_decisions, &pid);
    if (decision) {
        if (decision->priority > 5) {
            /* High priority: DataLoading, NcclCollective, OptimizerStep */
            slice = HIGH_PRIO_SLICE_NS;
            if (decision->preempt)
                flags |= SCX_ENQ_PREEMPT;
        } else if (decision->priority < 0) {
            /* Low priority: GpuCompute — yield CPU quickly */
            slice = LOW_PRIO_SLICE_NS;
        }
    }

    p->scx.slice = slice;
    scx_bpf_dispatch(p, SCX_DSQ_GLOBAL, slice, flags);
}

/*
 * dispatch — Called when a CPU needs work. Consume from global DSQ.
 */
void BPF_STRUCT_OPS(zernel_dispatch, s32 cpu, struct task_struct *prev)
{
    scx_bpf_consume(SCX_DSQ_GLOBAL);
}

/*
 * running — Task starts executing on a CPU.
 * Emit a ring buffer event so userspace can track phase timing.
 */
void BPF_STRUCT_OPS(zernel_running, struct task_struct *p)
{
    struct zernel_task_event *e;
    e = bpf_ringbuf_reserve(&task_events, sizeof(*e), 0);
    if (!e)
        return;

    e->pid = p->pid;
    e->event_type = 0;  /* running */
    e->timestamp_ns = bpf_ktime_get_ns();
    e->runtime_ns = 0;

    bpf_ringbuf_submit(e, 0);
}

/*
 * stopping — Task stops executing (preempted, blocked, or yielded).
 * Emit event with runtime duration for phase time accounting.
 */
void BPF_STRUCT_OPS(zernel_stopping, struct task_struct *p, bool runnable)
{
    struct zernel_task_event *e;
    e = bpf_ringbuf_reserve(&task_events, sizeof(*e), 0);
    if (!e)
        return;

    e->pid = p->pid;
    e->event_type = 1;  /* stopping */
    e->timestamp_ns = bpf_ktime_get_ns();
    e->runtime_ns = p->se.sum_exec_runtime;  /* total runtime so far */

    bpf_ringbuf_submit(e, 0);
}

/*
 * init_task — New task created. Initialize tracking state.
 */
s32 BPF_STRUCT_OPS(zernel_init_task, struct task_struct *p,
                   struct scx_init_task_args *args)
{
    struct zernel_task_event *e;
    e = bpf_ringbuf_reserve(&task_events, sizeof(*e), 0);
    if (e) {
        e->pid = p->pid;
        e->event_type = 2;  /* new task */
        e->timestamp_ns = bpf_ktime_get_ns();
        e->runtime_ns = 0;
        bpf_ringbuf_submit(e, 0);
    }

    return 0;
}

/*
 * init — Scheduler initialization. Called once when the scheduler is loaded.
 */
s32 BPF_STRUCT_OPS_SLEEPABLE(zernel_init)
{
    return 0;
}

/*
 * exit — Scheduler is being unloaded. Log the reason.
 */
void BPF_STRUCT_OPS(zernel_exit, struct scx_exit_info *ei)
{
    /* Userspace will log the exit reason via the skeleton */
}

/* ============================================================
 * Scheduler Registration
 * ============================================================ */

SCX_OPS_DEFINE(zernel_ops,
    .select_cpu     = (void *)zernel_select_cpu,
    .enqueue        = (void *)zernel_enqueue,
    .dispatch       = (void *)zernel_dispatch,
    .running        = (void *)zernel_running,
    .stopping       = (void *)zernel_stopping,
    .init_task      = (void *)zernel_init_task,
    .init           = (void *)zernel_init,
    .exit           = (void *)zernel_exit,
    .name           = "zernel",
);

#endif /* __BPF__ */

char _license[] SEC("license") = "GPL";
