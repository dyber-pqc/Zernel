// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel bench — ML benchmark suite

use anyhow::{Context, Result};
use clap::Subcommand;
use std::process::Command;
use std::time::Instant;

#[derive(Subcommand)]
pub enum BenchCommands {
    /// Run full benchmark suite
    All,
    /// Quick 5-minute smoke test
    Quick,
    /// GPU compute throughput
    Gpu,
    /// Multi-GPU NCCL communication bandwidth
    Nccl,
    /// Dataset loading throughput
    Dataloader {
        /// Path to dataset directory
        #[arg(default_value = ".")]
        path: String,
        /// Number of DataLoader workers
        #[arg(long, default_value = "4")]
        workers: u32,
    },
    /// GPU memory allocation benchmark
    Memory,
    /// End-to-end training benchmark
    E2e {
        /// Model to benchmark
        #[arg(long, default_value = "resnet50")]
        model: String,
        /// Number of iterations
        #[arg(long, default_value = "100")]
        iterations: u32,
    },
    /// Generate benchmark report
    Report,
}

fn run_python_bench(name: &str, code: &str) -> Result<(f64, String)> {
    println!("  Running: {name}...");
    let start = Instant::now();
    let output = Command::new("python3")
        .args(["-c", code])
        .output()
        .with_context(|| format!("failed to run benchmark: {name}"))?;
    let elapsed = start.elapsed().as_secs_f64();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        println!("    SKIP ({stderr})");
        return Ok((0.0, "SKIP".into()));
    }

    Ok((elapsed, stdout.trim().to_string()))
}

pub async fn run(cmd: BenchCommands) -> Result<()> {
    match cmd {
        BenchCommands::All | BenchCommands::Quick => {
            let quick = matches!(cmd, BenchCommands::Quick);
            println!(
                "Zernel ML Benchmark Suite {}",
                if quick { "(Quick)" } else { "(Full)" }
            );
            println!("{}", "=".repeat(60));
            println!();

            let mut results = Vec::new();

            // GPU Info
            if let Ok(output) = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ])
                .output()
            {
                let info = String::from_utf8_lossy(&output.stdout);
                println!("Hardware: {}", info.trim());
                println!();
            }

            // 1. GPU FLOPS
            println!("[1/5] GPU Compute Throughput");
            let iters = if quick { 100 } else { 1000 };
            let (_t, out) = run_python_bench(
                "matmul",
                &format!(
                    "import torch; torch.cuda.synchronize(); import time; \
                 a=torch.randn(4096,4096,device='cuda'); b=torch.randn(4096,4096,device='cuda'); \
                 torch.cuda.synchronize(); t0=time.time(); \
                 [torch.mm(a,b) for _ in range({iters})]; \
                 torch.cuda.synchronize(); t1=time.time(); \
                 tflops=2*4096**3*{iters}/((t1-t0)*1e12); \
                 print(f'{{tflops:.1f}} TFLOPS ({{(t1-t0):.2f}}s for {iters} matmuls)')"
                ),
            )?;
            results.push(("GPU Compute", out.clone()));
            println!("    {out}");
            println!();

            // 2. GPU Memory Bandwidth
            println!("[2/5] GPU Memory Bandwidth");
            let (_, out) = run_python_bench(
                "membw",
                "import torch, time; \
                 size=1024*1024*256; a=torch.randn(size,device='cuda'); \
                 torch.cuda.synchronize(); t0=time.time(); \
                 [a.clone() for _ in range(100)]; \
                 torch.cuda.synchronize(); t1=time.time(); \
                 bw=size*4*100/((t1-t0)*1e9); \
                 print(f'{bw:.0f} GB/s ({(t1-t0):.2f}s)')",
            )?;
            results.push(("Memory BW", out.clone()));
            println!("    {out}");
            println!();

            // 3. NCCL (multi-GPU only)
            println!("[3/5] NCCL Multi-GPU Communication");
            let (_, out) = run_python_bench(
                "nccl",
                "import torch, time; \
                 n=torch.cuda.device_count(); \
                 if n<2: print(f'SKIP (only {n} GPU)'); exit(); \
                 import torch.distributed as dist; \
                 print(f'{n} GPUs detected — NCCL bench requires torchrun')",
            )?;
            results.push(("NCCL", out.clone()));
            println!("    {out}");
            println!();

            // 4. DataLoader
            println!("[4/5] DataLoader Throughput");
            let (_, out) = run_python_bench(
                "dataloader",
                "import torch, time; from torch.utils.data import DataLoader, TensorDataset; \
                 ds=TensorDataset(torch.randn(10000,3,224,224),torch.randint(0,1000,(10000,))); \
                 dl=DataLoader(ds,batch_size=64,num_workers=4,pin_memory=True); \
                 t0=time.time(); \
                 for batch in dl: pass; \
                 t1=time.time(); \
                 print(f'{10000/(t1-t0):.0f} samples/s ({(t1-t0):.2f}s for 10K samples)')",
            )?;
            results.push(("DataLoader", out.clone()));
            println!("    {out}");
            println!();

            // 5. Training step
            println!("[5/5] Training Step Latency");
            let (_, out) = run_python_bench(
                "trainstep",
                "import torch, torch.nn as nn, time; \
                 model=nn.Linear(4096,4096).cuda(); opt=torch.optim.Adam(model.parameters()); \
                 x=torch.randn(256,4096,device='cuda'); \
                 torch.cuda.synchronize(); t0=time.time(); \
                 for _ in range(100): \
                     loss=model(x).sum(); loss.backward(); opt.step(); opt.zero_grad(); \
                 torch.cuda.synchronize(); t1=time.time(); \
                 print(f'{(t1-t0)/100*1000:.1f} ms/step ({(t1-t0):.2f}s for 100 steps)')",
            )?;
            results.push(("Train Step", out.clone()));
            println!("    {out}");

            // Summary
            println!();
            println!("{}", "=".repeat(60));
            println!("Summary");
            println!("{}", "-".repeat(60));
            for (name, result) in &results {
                println!("  {:<20} {}", name, result);
            }
            println!();
        }

        BenchCommands::Gpu => {
            println!("GPU Compute Benchmark");
            let (_, out) = run_python_bench(
                "matmul-full",
                "import torch, time; sizes=[1024,2048,4096,8192]; \
                 for s in sizes: \
                     a=torch.randn(s,s,device='cuda'); b=torch.randn(s,s,device='cuda'); \
                     torch.cuda.synchronize(); t0=time.time(); \
                     [torch.mm(a,b) for _ in range(100)]; \
                     torch.cuda.synchronize(); t1=time.time(); \
                     tflops=2*s**3*100/((t1-t0)*1e12); \
                     print(f'  {s}x{s}: {tflops:.1f} TFLOPS ({(t1-t0)/100*1000:.1f} ms/op)')",
            )?;
            println!("{out}");
        }

        BenchCommands::Nccl => {
            println!("NCCL Benchmark — requires multi-GPU + torchrun");
            println!("Run: torchrun --nproc_per_node=auto -m torch.distributed.run nccl_bench.py");
        }

        BenchCommands::Dataloader { path, workers } => {
            println!("DataLoader Benchmark (path: {path}, workers: {workers})");
            let (_, out) = run_python_bench(
                "dl-bench",
                &format!(
                "import torch, time, os; from torch.utils.data import DataLoader, TensorDataset; \
                 n=50000; ds=TensorDataset(torch.randn(n,3,224,224),torch.randint(0,1000,(n,))); \
                 for w in [0,1,2,4,{workers}]: \
                     dl=DataLoader(ds,batch_size=64,num_workers=w,pin_memory=True); \
                     t0=time.time(); \
                     for b in dl: pass; \
                     t1=time.time(); \
                     print(f'  workers={{w}}: {{n/(t1-t0):.0f}} samples/s')"
            ),
            )?;
            println!("{out}");
        }

        BenchCommands::Memory => {
            println!("GPU Memory Allocation Benchmark");
            let (_, out) = run_python_bench(
                "mem-bench",
                "import torch, time; \
                 sizes=[1,10,100,1000]; \
                 for mb in sizes: \
                     n=mb*1024*256; t0=time.time(); \
                     [torch.empty(n,device='cuda') for _ in range(100)]; \
                     torch.cuda.synchronize(); t1=time.time(); \
                     print(f'  {mb}MB alloc: {(t1-t0)/100*1e6:.0f} us/alloc')",
            )?;
            println!("{out}");
        }

        BenchCommands::E2e { model, iterations } => {
            println!("End-to-End Training Benchmark: {model} ({iterations} iterations)");
            let (_, out) = run_python_bench("e2e", &format!(
                "import torch, torchvision.models as m, time; \
                 model=getattr(m,'{model}')().cuda(); \
                 opt=torch.optim.SGD(model.parameters(),lr=0.01); \
                 x=torch.randn(32,3,224,224,device='cuda'); t=torch.randint(0,1000,(32,),device='cuda'); \
                 loss_fn=torch.nn.CrossEntropyLoss(); \
                 torch.cuda.synchronize(); t0=time.time(); \
                 for i in range({iterations}): \
                     out=model(x); loss=loss_fn(out,t); loss.backward(); opt.step(); opt.zero_grad(); \
                 torch.cuda.synchronize(); t1=time.time(); \
                 ips=32*{iterations}/(t1-t0); \
                 print(f'{model}: {{ips:.0f}} images/s ({{(t1-t0)/{iterations}*1000:.1f}} ms/step)')"
            ))?;
            println!("{out}");
        }

        BenchCommands::Report => {
            println!("Run: zernel bench all > benchmark-report.txt");
            println!("Full HTML report generation coming in a future release.");
        }
    }
    Ok(())
}
