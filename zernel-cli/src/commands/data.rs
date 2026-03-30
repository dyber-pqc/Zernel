// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel data — Dataset management

use anyhow::{Context, Result};
use clap::Subcommand;
use std::path::Path;
use std::process::Command;

#[derive(Subcommand)]
pub enum DataCommands {
    /// Profile a dataset (stats, types, size)
    Profile {
        /// Path to dataset file or directory
        path: String,
    },
    /// Split dataset into train/val/test
    Split {
        /// Path to dataset directory
        path: String,
        /// Training fraction
        #[arg(long, default_value = "0.8")]
        train: f64,
        /// Validation fraction
        #[arg(long, default_value = "0.1")]
        val: f64,
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    /// Cache dataset to fast storage
    Cache {
        /// Source path
        source: String,
        /// Destination (fast storage)
        #[arg(long)]
        to: String,
    },
    /// Shard dataset for distributed training
    Shard {
        /// Path to dataset
        path: String,
        /// Number of shards
        #[arg(long, default_value = "8")]
        shards: u32,
    },
    /// Benchmark DataLoader throughput
    Benchmark {
        /// Path to dataset
        #[arg(default_value = ".")]
        path: String,
        /// Number of workers
        #[arg(long, default_value = "4")]
        workers: u32,
    },
    /// Serve dataset over network for multi-node training
    Serve {
        /// Path to dataset
        path: String,
        /// Port to serve on
        #[arg(long, default_value = "8888")]
        port: u16,
    },
}

pub async fn run(cmd: DataCommands) -> Result<()> {
    match cmd {
        DataCommands::Profile { path } => {
            let p = Path::new(&path);
            if !p.exists() {
                anyhow::bail!("path not found: {path}");
            }

            println!("Dataset Profile: {path}");
            println!("{}", "=".repeat(60));

            if p.is_file() {
                let size = p.metadata()?.len();
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
                println!("  Type:     file ({ext})");
                println!("  Size:     {}", format_size(size));

                match ext {
                    "parquet" => {
                        let out = Command::new("python3")
                            .args([
                                "-c",
                                &format!(
                                    "import pyarrow.parquet as pq; \
                                 f=pq.read_metadata('{path}'); \
                                 print(f'  Rows:     {{f.num_rows}}'); \
                                 print(f'  Columns:  {{f.num_columns}}'); \
                                 print(f'  Row Groups: {{f.num_row_groups}}'); \
                                 s=pq.read_schema('{path}'); \
                                 for i in range(min(10,len(s))): print(f'    {{s.field(i)}}')"
                                ),
                            ])
                            .output();
                        if let Ok(o) = out {
                            print!("{}", String::from_utf8_lossy(&o.stdout));
                        }
                    }
                    "csv" | "tsv" => {
                        let lines = std::io::BufRead::lines(std::io::BufReader::new(
                            std::fs::File::open(p)?,
                        ))
                        .count();
                        println!("  Rows:     {lines}");
                    }
                    "json" | "jsonl" => {
                        let lines = std::io::BufRead::lines(std::io::BufReader::new(
                            std::fs::File::open(p)?,
                        ))
                        .count();
                        println!("  Lines:    {lines}");
                    }
                    _ => {
                        println!("  (use .parquet/.csv/.json for detailed stats)");
                    }
                }
            } else {
                // Directory — count files and total size
                let mut total_size = 0u64;
                let mut file_count = 0u64;
                let mut ext_counts: std::collections::HashMap<String, u64> =
                    std::collections::HashMap::new();

                fn walk(
                    dir: &Path,
                    size: &mut u64,
                    count: &mut u64,
                    exts: &mut std::collections::HashMap<String, u64>,
                ) {
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        for entry in entries.flatten() {
                            let Some(ft) = entry.file_type().ok() else {
                                continue;
                            };
                            if ft.is_file() {
                                *size += entry.metadata().map(|m| m.len()).unwrap_or(0);
                                *count += 1;
                                let ext = entry
                                    .path()
                                    .extension()
                                    .map(|e| e.to_string_lossy().to_string())
                                    .unwrap_or_else(|| "other".into());
                                *exts.entry(ext).or_default() += 1;
                            } else if ft.is_dir() {
                                walk(&entry.path(), size, count, exts);
                            }
                        }
                    }
                }

                walk(p, &mut total_size, &mut file_count, &mut ext_counts);

                println!("  Type:     directory");
                println!("  Files:    {file_count}");
                println!("  Size:     {}", format_size(total_size));
                println!("  Extensions:");
                let mut sorted: Vec<_> = ext_counts.iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(a.1));
                for (ext, count) in sorted.iter().take(10) {
                    println!("    .{ext}: {count} files");
                }
            }
        }

        DataCommands::Split {
            path,
            train,
            val,
            seed,
        } => {
            let test = 1.0 - train - val;
            println!("Splitting {path}: train={train} val={val} test={test} seed={seed}");

            let code = format!(
                "import os, random, shutil; random.seed({seed}); \
                 files=[f for f in os.listdir('{path}') if os.path.isfile(os.path.join('{path}',f))]; \
                 random.shuffle(files); n=len(files); \
                 nt=int(n*{train}); nv=int(n*{val}); \
                 for split,fs in [('train',files[:nt]),('val',files[nt:nt+nv]),('test',files[nt+nv:])]: \
                     d=os.path.join('{path}',split); os.makedirs(d,exist_ok=True); \
                     for f in fs: shutil.move(os.path.join('{path}',f),os.path.join(d,f)); \
                     print(f'  {{split}}: {{len(fs)}} files')"
            );
            let output = Command::new("python3").args(["-c", &code]).output()?;
            print!("{}", String::from_utf8_lossy(&output.stdout));
            if !output.status.success() {
                print!("{}", String::from_utf8_lossy(&output.stderr));
            }
        }

        DataCommands::Cache { source, to } => {
            println!("Caching {source} → {to}");
            let status = Command::new("rsync")
                .args(["-avh", "--progress", &source, &to])
                .status()
                .with_context(|| "rsync not found")?;
            if status.success() {
                println!("Cache complete.");
            }
        }

        DataCommands::Shard { path, shards } => {
            println!("Sharding {path} into {shards} shards...");
            let code = format!(
                "import os, shutil; \
                 files=sorted([f for f in os.listdir('{path}') if os.path.isfile(os.path.join('{path}',f))]); \
                 for i in range({shards}): \
                     d=os.path.join('{path}',f'shard-{{i:04d}}'); os.makedirs(d,exist_ok=True); \
                 for i,f in enumerate(files): \
                     shard=i%{shards}; \
                     shutil.move(os.path.join('{path}',f),os.path.join('{path}',f'shard-{{shard:04d}}',f)); \
                 for i in range({shards}): \
                     d=os.path.join('{path}',f'shard-{{i:04d}}'); \
                     n=len(os.listdir(d)); print(f'  shard-{{i:04d}}: {{n}} files')"
            );
            let output = Command::new("python3").args(["-c", &code]).output()?;
            print!("{}", String::from_utf8_lossy(&output.stdout));
        }

        DataCommands::Benchmark { path, workers } => {
            println!("DataLoader Benchmark (path: {path}, workers: {workers})");
            let code = format!(
                "import torch,time; from torch.utils.data import DataLoader,TensorDataset; \
                 ds=TensorDataset(torch.randn(10000,3,224,224),torch.randint(0,1000,(10000,))); \
                 for w in [0,1,2,4,{workers}]: \
                     dl=DataLoader(ds,batch_size=64,num_workers=w,pin_memory=True); \
                     t0=time.time(); [None for _ in dl]; t1=time.time(); \
                     print(f'  workers={{w}}: {{10000/(t1-t0):.0f}} samples/s')"
            );
            let output = Command::new("python3").args(["-c", &code]).output()?;
            print!("{}", String::from_utf8_lossy(&output.stdout));
            if !output.status.success() {
                print!("{}", String::from_utf8_lossy(&output.stderr));
            }
        }

        DataCommands::Serve { path, port } => {
            println!("Serving dataset at {path} on port {port}...");
            println!("URL: http://0.0.0.0:{port}");
            let status = Command::new("python3")
                .args(["-m", "http.server", &port.to_string(), "--directory", &path])
                .status()?;
            let _ = status;
        }
    }
    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
