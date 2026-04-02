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
            .clang("clang-16")
            .clang_args([
                format!("-I{}", bpf_dir.display()),
                "-D__BPF__".to_string(),
                "-target".to_string(),
                "bpf".to_string(),
            ])
            .build_and_generate(&out)
            .unwrap_or_else(|e| {
                panic!("failed to build sched_ext BPF skeleton: {e}");
            });

        println!("cargo:rerun-if-changed={}", src.display());
        println!("cargo:rerun-if-changed=src/bpf/zernel_maps.h");
    }
}
