/* Copyright (C) 2026 Dyber, Inc. — GPL-2.0 */
/*
 * Zernel Distributed Sync Probe
 *
 * Instruments futex_wait to measure rank synchronization overhead
 * in distributed training. High futex wait times indicate straggler
 * ranks or gradient accumulation bottlenecks.
 *
 * Attach: kprobe/kretprobe on do_futex or futex_wait_queue
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
} dist_sync_events SEC(".maps");

/* Entry timestamp per TID */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TRACKED_PIDS * 16);
    __type(key, __u64);    /* pid_tgid */
    __type(value, __u64);  /* start timestamp_ns */
} futex_start SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TRACKED_PIDS);
    __type(key, __u32);
    __type(value, __u8);
} tracked_pids SEC(".maps");

/*
 * kprobe on futex_wait (kernel internal, exact symbol varies by version)
 *
 * We use do_futex as a more stable entry point.
 * FUTEX_WAIT = 0, FUTEX_WAIT_BITSET = 9
 */
#ifdef __BPF__
SEC("kprobe/do_futex")
int BPF_KPROBE(trace_futex_entry, __u32 *uaddr, int op)
{
    /* Only trace FUTEX_WAIT and FUTEX_WAIT_BITSET */
    int cmd = op & 0x7f;  /* mask out FUTEX_PRIVATE_FLAG */
    if (cmd != 0 && cmd != 9)
        return 0;

    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    if (!bpf_map_lookup_elem(&tracked_pids, &pid))
        return 0;

    __u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&futex_start, &pid_tgid, &ts, BPF_ANY);
    return 0;
}

SEC("kretprobe/do_futex")
int BPF_KRETPROBE(trace_futex_return)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    __u32 tid = (__u32)pid_tgid;

    __u64 *start_ts = bpf_map_lookup_elem(&futex_start, &pid_tgid);
    if (!start_ts)
        return 0;

    __u64 end_ts = bpf_ktime_get_ns();
    __u64 wait_ns = end_ts - *start_ts;

    bpf_map_delete_elem(&futex_start, &pid_tgid);

    /* Only report waits > 1us (filter out spurious wakeups) */
    if (wait_ns < 1000)
        return 0;

    struct zernel_dist_sync_event *e;
    e = bpf_ringbuf_reserve(&dist_sync_events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->pid = pid;
    e->tid = tid;
    e->wait_ns = wait_ns;
    e->timestamp_ns = end_ts;

    bpf_ringbuf_submit(e, 0);
    return 0;
}
#endif /* __BPF__ */

char _license[] SEC("license") = "GPL";
