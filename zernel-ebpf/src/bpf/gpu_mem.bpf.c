/* Copyright (C) 2026 Dyber, Inc. — GPL-2.0 */
/*
 * Zernel GPU Memory Probe
 *
 * Tracks CUDA memory allocation and deallocation by attaching uprobes
 * to cuMemAlloc_v2 and cuMemFree_v2 in libcuda.so.
 *
 * Reports per-process, per-GPU memory usage to the userspace daemon
 * via a BPF ring buffer.
 *
 * Requires: Linux 6.12+, CONFIG_BPF=y, CONFIG_BPF_SYSCALL=y
 * Attach:   uprobe on libcuda.so::cuMemAlloc_v2 / cuMemFree_v2
 */

#ifdef __BPF__
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#endif

#include "common.h"

/* Ring buffer for sending events to userspace */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, RINGBUF_SIZE);
} gpu_mem_events SEC(".maps");

/* Per-PID:GPU running memory total */
struct mem_key {
    __u32 pid;
    __u32 gpu_id;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TRACKED_PIDS);
    __type(key, struct mem_key);
    __type(value, __u64);
} gpu_mem_usage SEC(".maps");

/* PID filter — only trace PIDs in this set */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TRACKED_PIDS);
    __type(key, __u32);
    __type(value, __u8);
} tracked_pids SEC(".maps");

/*
 * Uprobe on cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
 *
 * Arguments:
 *   arg0 (ctx->ax / PT_REGS_PARM1): CUdeviceptr *dptr
 *   arg1 (ctx->dx / PT_REGS_PARM2): size_t bytesize
 */
#ifdef __BPF__
SEC("uprobe/cuMemAlloc_v2")
int BPF_UPROBE(trace_cu_mem_alloc, void *dptr, __u64 bytesize)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;

    /* Check PID filter */
    if (!bpf_map_lookup_elem(&tracked_pids, &pid))
        return 0;

    struct mem_key key = { .pid = pid, .gpu_id = 0 };
    __u64 *current = bpf_map_lookup_elem(&gpu_mem_usage, &key);
    __u64 new_total = current ? *current + bytesize : bytesize;
    bpf_map_update_elem(&gpu_mem_usage, &key, &new_total, BPF_ANY);

    /* Emit event */
    struct zernel_gpu_mem_event *e;
    e = bpf_ringbuf_reserve(&gpu_mem_events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->pid = pid;
    e->gpu_id = 0;
    e->alloc_bytes = bytesize;
    e->free_bytes = 0;
    e->total_usage = new_total;
    e->timestamp_ns = bpf_ktime_get_ns();

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Uprobe on cuMemFree_v2(CUdeviceptr dptr)
 *
 * We don't know the size being freed from the argument alone.
 * In practice, we'd need to track the dptr→size mapping in a
 * BPF hash map at allocation time. For now, we emit a free event
 * and let userspace handle the bookkeeping.
 */
SEC("uprobe/cuMemFree_v2")
int BPF_UPROBE(trace_cu_mem_free, __u64 dptr)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;

    if (!bpf_map_lookup_elem(&tracked_pids, &pid))
        return 0;

    struct zernel_gpu_mem_event *e;
    e = bpf_ringbuf_reserve(&gpu_mem_events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->pid = pid;
    e->gpu_id = 0;
    e->alloc_bytes = 0;
    e->free_bytes = 0;  /* size unknown from free() arg */
    e->total_usage = 0;
    e->timestamp_ns = bpf_ktime_get_ns();

    bpf_ringbuf_submit(e, 0);
    return 0;
}
#endif /* __BPF__ */

char _license[] SEC("license") = "GPL";
