// Copyright (C) 2026 Dyber, Inc. — GPL-2.0
//
// Build script for zernel-ebpf.
//
// When the `bpf` feature is enabled, compiles BPF C sources into
// skeleton .rs files using libbpf-cargo. When disabled (default),
// this is a no-op — allowing cross-platform development.

fn main() {
    #[cfg(feature = "bpf")]
    {
        use libbpf_cargo::SkeletonBuilder;
        use std::path::Path;

        let bpf_dir = Path::new("src/bpf");
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");

        let probes = ["gpu_mem", "cuda_trace", "nccl", "dataload", "dist_sync"];

        for probe in &probes {
            let src = bpf_dir.join(format!("{probe}.bpf.c"));
            let out = Path::new(&out_dir).join(format!("{probe}.skel.rs"));

            SkeletonBuilder::new()
                .source(&src)
                .clang_args(format!("-I{}", bpf_dir.display()))
                .build_and_generate(&out)
                .unwrap_or_else(|e| {
                    panic!("failed to build BPF skeleton for {probe}: {e}");
                });

            println!("cargo:rerun-if-changed={}", src.display());
        }

        println!("cargo:rerun-if-changed=src/bpf/common.h");
    }
}
