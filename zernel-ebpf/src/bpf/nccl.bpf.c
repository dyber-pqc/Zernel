/* Copyright (C) 2026 Dyber, Inc. — GPL-2.0 */
/*
 * Zernel NCCL Collective Trace
 *
 * Measures duration of NCCL collective operations by attaching
 * uprobe/uretprobe pairs on libnccl.so functions:
 *   ncclAllReduce, ncclBroadcast, ncclAllGather, ncclReduceScatter
 *
 * Reports operation type, data size, duration, and rank info.
 *
 * Attach: uprobe + uretprobe on libnccl.so::{ncclAllReduce,...}
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
} nccl_events SEC(".maps");

/* Entry timestamp + metadata per TID */
struct nccl_entry {
    __u64 start_ns;
    __u8  op;
    __u64 size_bytes;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TRACKED_PIDS);
    __type(key, __u64);                   /* pid_tgid */
    __type(value, struct nccl_entry);
} nccl_start SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TRACKED_PIDS);
    __type(key, __u32);
    __type(value, __u8);
} tracked_pids SEC(".maps");

/*
 * Common entry handler — records timestamp and operation type.
 * Called by each per-op uprobe.
 */
#ifdef __BPF__
static __always_inline int nccl_entry_common(__u8 op, __u64 count, __u64 datatype_size)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    if (!bpf_map_lookup_elem(&tracked_pids, &pid))
        return 0;

    struct nccl_entry entry = {
        .start_ns = bpf_ktime_get_ns(),
        .op = op,
        .size_bytes = count * datatype_size,
    };
    bpf_map_update_elem(&nccl_start, &pid_tgid, &entry, BPF_ANY);
    return 0;
}

static __always_inline int nccl_return_common(void)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    struct nccl_entry *entry = bpf_map_lookup_elem(&nccl_start, &pid_tgid);
    if (!entry)
        return 0;

    __u64 end_ns = bpf_ktime_get_ns();

    struct zernel_nccl_event *e;
    e = bpf_ringbuf_reserve(&nccl_events, sizeof(*e), 0);
    if (!e) {
        bpf_map_delete_elem(&nccl_start, &pid_tgid);
        return 0;
    }

    e->pid = pid;
    e->op = entry->op;
    e->size_bytes = entry->size_bytes;
    e->start_ns = entry->start_ns;
    e->duration_ns = end_ns - entry->start_ns;
    e->rank = 0;       /* TODO: read from ncclComm struct */
    e->num_ranks = 0;  /* TODO: read from ncclComm struct */

    bpf_ringbuf_submit(e, 0);
    bpf_map_delete_elem(&nccl_start, &pid_tgid);
    return 0;
}

/* ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream) */
SEC("uprobe/ncclAllReduce")
int BPF_UPROBE(trace_nccl_allreduce, void *send, void *recv, __u64 count)
{
    return nccl_entry_common(NCCL_OP_ALL_REDUCE, count, 4); /* assume float32 */
}

SEC("uretprobe/ncclAllReduce")
int BPF_URETPROBE(trace_nccl_allreduce_ret)
{
    return nccl_return_common();
}

SEC("uprobe/ncclBroadcast")
int BPF_UPROBE(trace_nccl_broadcast, void *send, void *recv, __u64 count)
{
    return nccl_entry_common(NCCL_OP_BROADCAST, count, 4);
}

SEC("uretprobe/ncclBroadcast")
int BPF_URETPROBE(trace_nccl_broadcast_ret)
{
    return nccl_return_common();
}

SEC("uprobe/ncclAllGather")
int BPF_UPROBE(trace_nccl_allgather, void *send, void *recv, __u64 count)
{
    return nccl_entry_common(NCCL_OP_ALL_GATHER, count, 4);
}

SEC("uretprobe/ncclAllGather")
int BPF_URETPROBE(trace_nccl_allgather_ret)
{
    return nccl_return_common();
}

SEC("uprobe/ncclReduceScatter")
int BPF_UPROBE(trace_nccl_reducescatter, void *send, void *recv, __u64 count)
{
    return nccl_entry_common(NCCL_OP_REDUCE_SCATTER, count, 4);
}

SEC("uretprobe/ncclReduceScatter")
int BPF_URETPROBE(trace_nccl_reducescatter_ret)
{
    return nccl_return_common();
}
#endif /* __BPF__ */

char _license[] SEC("license") = "GPL";
