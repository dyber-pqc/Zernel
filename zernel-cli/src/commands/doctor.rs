// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::Result;

/// Diagnose the Zernel environment — check GPU drivers, CUDA, frameworks, etc.
pub async fn run() -> Result<()> {
    println!("Zernel Doctor");
    println!("=============");
    println!();

    // TODO: Run actual checks via Command::new()
    let checks = [
        ("Linux kernel", "check kernel version >= 6.12"),
        ("sched_ext", "check CONFIG_SCHED_CLASS_EXT=y"),
        ("NVIDIA driver", "nvidia-smi"),
        ("CUDA toolkit", "nvcc --version"),
        ("cuDNN", "check libcudnn.so"),
        ("NCCL", "check libnccl.so"),
        ("Python", "python3 --version"),
        ("PyTorch", "python3 -c 'import torch; print(torch.__version__)'"),
        ("PyTorch CUDA", "python3 -c 'import torch; print(torch.cuda.is_available())'"),
        ("zerneld", "check if zerneld is running on port 9091"),
        ("Huge pages", "check vm.nr_hugepages"),
        ("NUMA", "check kernel.numa_balancing"),
    ];

    for (name, description) in checks {
        println!("  [ ] {name}: {description}");
    }

    println!();
    println!("(checks not yet implemented — showing planned diagnostics)");

    Ok(())
}
