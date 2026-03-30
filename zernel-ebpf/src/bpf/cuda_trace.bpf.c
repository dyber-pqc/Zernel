/* Copyright (C) 2026 Dyber, Inc. — GPL-2.0 */
/*
 * Zernel CUDA Kernel Launch Trace
 *
 * Measures CUDA kernel launch latency by attaching uprobe/uretprobe
 * on cuLaunchKernel in libcuda.so. The latency is the time between
 * the function entry and return — this captures driver overhead,
 * command buffer submission, and any synchronous waits.
 *
 * Attach: uprobe + uretprobe on libcuda.so::cuLaunchKernel
 */

#ifdef __BPF__
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#endif

#include "common.h"

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, RINGBUF_SIZE);
} cuda_events SEC(".maps");

/* Temporary storage for launch entry timestamp (per-TID) */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TRACKED_PIDS);
    __type(key, __u64);    /* pid_tgid */
    __type(value, __u64);  /* entry timestamp_ns */
} cuda_launch_start SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TRACKED_PIDS);
    __type(key, __u32);
    __type(value, __u8);
} tracked_pids SEC(".maps");

/*
 * cuLaunchKernel(CUfunction f, gridDimX, gridDimY, gridDimZ,
 *                blockDimX, blockDimY, blockDimZ,
 *                sharedMemBytes, hStream, kernelParams, extra)
 *
 * We record the entry timestamp keyed by pid_tgid.
 */
#ifdef __BPF__
SEC("uprobe/cuLaunchKernel")
int BPF_UPROBE(trace_cuda_launch_entry, void *func)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    if (!bpf_map_lookup_elem(&tracked_pids, &pid))
        return 0;

    __u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&cuda_launch_start, &pid_tgid, &ts, BPF_ANY);
    return 0;
}

SEC("uretprobe/cuLaunchKernel")
int BPF_URETPROBE(trace_cuda_launch_return)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    __u64 *start_ts = bpf_map_lookup_elem(&cuda_launch_start, &pid_tgid);
    if (!start_ts)
        return 0;

    __u64 end_ts = bpf_ktime_get_ns();
    __u64 latency = end_ts - *start_ts;

    bpf_map_delete_elem(&cuda_launch_start, &pid_tgid);

    struct zernel_cuda_event *e;
    e = bpf_ringbuf_reserve(&cuda_events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->pid = pid;
    e->kernel_hash = 0;   /* TODO: hash kernel function name */
    e->launch_ns = *start_ts;
    e->return_ns = end_ts;
    e->latency_ns = latency;

    bpf_ringbuf_submit(e, 0);
    return 0;
}
#endif /* __BPF__ */

char _license[] SEC("license") = "GPL";
