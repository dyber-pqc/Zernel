/* Copyright (C) 2026 Dyber, Inc. — GPL-2.0 */
/* Zernel eBPF observability — shared definitions */

#ifndef __ZERNEL_COMMON_H
#define __ZERNEL_COMMON_H

/* Event types shared between BPF and userspace.
 * These structs are serialized directly from BPF ring buffers
 * and deserialized in the Rust consumer code via raw pointer casts.
 *
 * IMPORTANT: These must stay in sync with the Rust event structs
 * in zernel-ebpf/src/consumers/*.rs. Field order, sizes, and
 * alignment must match exactly.
 */

/* ============================================================
 * GPU Memory Events (gpu_mem.bpf.c)
 * Tracks CUDA memory allocation/deallocation via uprobes on libcuda.so
 * ============================================================ */
struct zernel_gpu_mem_event {
    __u32 pid;
    __u32 gpu_id;
    __u64 alloc_bytes;    /* bytes allocated in this call (0 if free) */
    __u64 free_bytes;     /* bytes freed in this call (0 if alloc) */
    __u64 total_usage;    /* running total for this pid:gpu_id */
    __u64 timestamp_ns;
};

/* ============================================================
 * CUDA Kernel Launch Events (cuda_trace.bpf.c)
 * Instruments cuLaunchKernel via uprobe/uretprobe on libcuda.so
 * ============================================================ */
struct zernel_cuda_event {
    __u32 pid;
    __u64 kernel_hash;    /* FNV-1a hash of kernel function name */
    __u64 launch_ns;      /* timestamp at uprobe entry */
    __u64 return_ns;      /* timestamp at uretprobe return */
    __u64 latency_ns;     /* return_ns - launch_ns */
};

/* ============================================================
 * NCCL Collective Events (nccl.bpf.c)
 * Instruments ncclAllReduce/Broadcast/AllGather/ReduceScatter
 * ============================================================ */
#define NCCL_OP_ALL_REDUCE       0
#define NCCL_OP_BROADCAST        1
#define NCCL_OP_ALL_GATHER       2
#define NCCL_OP_REDUCE_SCATTER   3

struct zernel_nccl_event {
    __u32 pid;
    __u8  op;             /* NCCL_OP_* constant */
    __u8  _pad[3];
    __u64 size_bytes;     /* data size in collective */
    __u64 start_ns;
    __u64 duration_ns;
    __u32 rank;
    __u32 num_ranks;
};

/* ============================================================
 * DataLoader I/O Events (dataload.bpf.c)
 * Instruments read/io_uring syscalls from DataLoader worker threads
 * ============================================================ */
struct zernel_dataload_event {
    __u32 pid;
    __u32 tid;            /* thread ID (worker identification) */
    __u64 read_bytes;
    __u64 latency_ns;
    __u8  io_type;        /* 0 = read(), 1 = io_uring */
    __u8  _pad[7];
};

/* ============================================================
 * Distributed Sync Events (dist_sync.bpf.c)
 * Instruments futex_wait for training rank synchronization
 * ============================================================ */
struct zernel_dist_sync_event {
    __u32 pid;
    __u32 tid;
    __u64 wait_ns;        /* time spent in futex_wait */
    __u64 timestamp_ns;
};

/* ============================================================
 * Shared maps and helpers
 * ============================================================ */

/* Maximum number of PIDs to track */
#define MAX_TRACKED_PIDS  4096

/* Ring buffer size (must be power of 2, in pages) */
#define RINGBUF_SIZE      (256 * 1024)  /* 256KB */

#endif /* __ZERNEL_COMMON_H */
