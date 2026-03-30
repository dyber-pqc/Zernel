/* Copyright (C) 2026 Dyber, Inc. — GPL-2.0 */
/*
 * Zernel DataLoader I/O Probe
 *
 * Instruments read() and io_uring operations from DataLoader worker
 * threads to measure dataset loading throughput and latency.
 *
 * Only traces PIDs registered in the tracked_pids map.
 *
 * Attach: kprobe/kretprobe on vfs_read (for standard read())
 *         tracepoint/io_uring/io_uring_complete (for io_uring)
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
} dataload_events SEC(".maps");

/* Entry timestamp per TID */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TRACKED_PIDS * 16); /* many worker threads */
    __type(key, __u64);    /* pid_tgid */
    __type(value, __u64);  /* start timestamp_ns */
} read_start SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TRACKED_PIDS);
    __type(key, __u32);
    __type(value, __u8);
} tracked_pids SEC(".maps");

/*
 * kprobe on vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)
 */
#ifdef __BPF__
SEC("kprobe/vfs_read")
int BPF_KPROBE(trace_vfs_read_entry, void *file, void *buf, __u64 count)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    if (!bpf_map_lookup_elem(&tracked_pids, &pid))
        return 0;

    /* Only trace reads > 4KB (skip small metadata reads) */
    if (count < 4096)
        return 0;

    __u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&read_start, &pid_tgid, &ts, BPF_ANY);
    return 0;
}

SEC("kretprobe/vfs_read")
int BPF_KRETPROBE(trace_vfs_read_return, long ret)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    __u32 tid = (__u32)pid_tgid;

    __u64 *start_ts = bpf_map_lookup_elem(&read_start, &pid_tgid);
    if (!start_ts)
        return 0;

    __u64 end_ts = bpf_ktime_get_ns();
    __u64 latency = end_ts - *start_ts;

    bpf_map_delete_elem(&read_start, &pid_tgid);

    /* Only report if read returned positive bytes */
    if (ret <= 0)
        return 0;

    struct zernel_dataload_event *e;
    e = bpf_ringbuf_reserve(&dataload_events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->pid = pid;
    e->tid = tid;
    e->read_bytes = (__u64)ret;
    e->latency_ns = latency;
    e->io_type = 0;  /* standard read() */

    bpf_ringbuf_submit(e, 0);
    return 0;
}
#endif /* __BPF__ */

char _license[] SEC("license") = "GPL";
