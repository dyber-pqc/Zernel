/* Copyright (C) 2026 Dyber, Inc. — GPL-2.0 */
/*
 * Zernel sched_ext BPF scheduler
 *
 * This BPF program implements the kernel-side scheduling hooks for the
 * Zernel ML-aware scheduler. It reads scheduling decisions from BPF maps
 * (written by the userspace Rust component) and applies them at scheduling
 * decision points.
 *
 * Requires: Linux 6.12+ with CONFIG_SCHED_CLASS_EXT=y
 *
 * This file is compiled to BPF bytecode by clang and loaded via libbpf.
 * It is NOT compiled as part of the normal Cargo build on non-Linux platforms.
 */

// NOTE: This is a reference skeleton. Full implementation requires:
// - vmlinux.h (generated from the target kernel)
// - scx BPF helpers
// - Full sched_ext ops implementation
//
// The actual BPF build will be wired up when targeting Linux.

#include "zernel_maps.h"

/*
 * BPF maps — shared between kernel BPF and userspace Rust.
 *
 * task_states: per-PID task state (written by userspace phase detector)
 * sched_decisions: per-PID scheduling decisions (written by userspace)
 *
 * struct {
 *     __uint(type, BPF_MAP_TYPE_HASH);
 *     __uint(max_entries, 65536);
 *     __type(key, __u32);                        // pid
 *     __type(value, struct zernel_task_state);
 * } task_states SEC(".maps");
 *
 * struct {
 *     __uint(type, BPF_MAP_TYPE_HASH);
 *     __uint(max_entries, 65536);
 *     __type(key, __u32);                        // pid
 *     __type(value, struct zernel_sched_decision);
 * } sched_decisions SEC(".maps");
 */

/*
 * sched_ext ops to implement:
 *
 * s32 BPF_STRUCT_OPS(zernel_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
 * void BPF_STRUCT_OPS(zernel_enqueue, struct task_struct *p, u64 enq_flags)
 * void BPF_STRUCT_OPS(zernel_dispatch, s32 cpu, struct task_struct *prev)
 * void BPF_STRUCT_OPS(zernel_running, struct task_struct *p)
 * void BPF_STRUCT_OPS(zernel_stopping, struct task_struct *p, bool runnable)
 * s32 BPF_STRUCT_OPS(zernel_init_task, struct task_struct *p, struct scx_init_task_args *args)
 * s32 BPF_STRUCT_OPS_SLEEPABLE(zernel_init)
 * void BPF_STRUCT_OPS(zernel_exit, struct scx_exit_info *ei)
 */

char _license[] __attribute__((section("license"))) = "GPL";
