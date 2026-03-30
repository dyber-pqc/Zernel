// Copyright (C) 2026 Dyber, Inc. — GPL-2.0
//
// Build script for zernel-scheduler.
// Compiles the sched_ext BPF program when the `bpf` feature is enabled.

fn main() {
    #[cfg(feature = "bpf")]
    {
        use libbpf_cargo::SkeletonBuilder;
        use std::path::Path;

        let bpf_dir = Path::new("src/bpf");
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        let src = bpf_dir.join("zernel_sched.bpf.c");
        let out = Path::new(&out_dir).join("zernel_sched.skel.rs");

        SkeletonBuilder::new()
            .source(&src)
            .clang_args(format!("-I{}", bpf_dir.display()))
            .build_and_generate(&out)
            .unwrap_or_else(|e| {
                panic!("failed to build sched_ext BPF skeleton: {e}");
            });

        println!("cargo:rerun-if-changed={}", src.display());
        println!("cargo:rerun-if-changed=src/bpf/zernel_maps.h");
    }
}
