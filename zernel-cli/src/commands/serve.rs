// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel serve — Unified inference server

use anyhow::{Context, Result};
use clap::Subcommand;
use std::process::Command;

#[derive(Subcommand)]
pub enum ServeCommands {
    /// Start serving a model
    Start {
        /// Path to model or model name
        model: String,
        /// Inference engine (auto, vllm, trt, onnx)
        #[arg(long, default_value = "auto")]
        engine: String,
        /// Port to serve on
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Number of GPU replicas
        #[arg(long, default_value = "1")]
        replicas: u32,
        /// Quantization (none, int8, int4)
        #[arg(long, default_value = "none")]
        quantize: String,
    },
    /// List running inference servers
    List,
    /// Stop an inference server
    Stop {
        /// Model name or port
        model: String,
    },
    /// Show server logs
    Logs {
        /// Model name
        model: String,
    },
    /// Load test an inference endpoint
    Benchmark {
        /// URL to benchmark
        #[arg(default_value = "http://localhost:8080")]
        url: String,
        /// Queries per second
        #[arg(long, default_value = "10")]
        qps: u32,
        /// Duration in seconds
        #[arg(long, default_value = "30")]
        duration: u32,
    },
}

fn detect_engine(model_path: &str) -> &'static str {
    let path = std::path::Path::new(model_path);

    // Check for engine-specific files
    if path.join("config.json").exists() {
        return "vllm"; // HuggingFace model
    }
    if path.extension().map(|e| e == "onnx").unwrap_or(false) {
        return "onnx";
    }
    if path
        .extension()
        .map(|e| e == "engine" || e == "plan")
        .unwrap_or(false)
    {
        return "trt";
    }

    // Default to vLLM (most versatile)
    "vllm"
}

pub async fn run(cmd: ServeCommands) -> Result<()> {
    match cmd {
        ServeCommands::Start {
            model,
            engine,
            port,
            replicas,
            quantize,
        } => {
            let selected_engine = if engine == "auto" {
                detect_engine(&model)
            } else {
                engine.as_str()
            };

            println!("Zernel Serve");
            println!("  Model:    {model}");
            println!("  Engine:   {selected_engine}");
            println!("  Port:     {port}");
            println!("  Replicas: {replicas}");
            println!("  Quantize: {quantize}");
            println!();

            match selected_engine {
                "vllm" => {
                    let mut args = vec![
                        "-m".into(),
                        "vllm.entrypoints.openai.api_server".into(),
                        "--model".into(),
                        model.clone(),
                        "--port".into(),
                        port.to_string(),
                    ];

                    if replicas > 1 {
                        args.extend(["--tensor-parallel-size".into(), replicas.to_string()]);
                    }

                    if quantize != "none" {
                        args.extend(["--quantization".into(), quantize]);
                    }

                    println!("Starting vLLM server...");
                    println!("  URL: http://localhost:{port}/v1");
                    println!("  Docs: http://localhost:{port}/docs");
                    println!("  Press Ctrl+C to stop");
                    println!();

                    let status = tokio::process::Command::new("python3")
                        .args(&args)
                        .status()
                        .await
                        .with_context(|| "failed to start vLLM — install with: pip install vllm")?;

                    if !status.success() {
                        anyhow::bail!("vLLM exited with code {}", status.code().unwrap_or(-1));
                    }
                }

                "trt" => {
                    println!("Starting TensorRT server...");
                    let status = tokio::process::Command::new("tritonserver")
                        .args([
                            "--model-repository",
                            &model,
                            "--http-port",
                            &port.to_string(),
                        ])
                        .status()
                        .await
                        .with_context(|| "tritonserver not found — install NVIDIA Triton")?;
                    let _ = status;
                }

                "onnx" => {
                    println!("Starting ONNX Runtime server...");
                    let code = format!(
                        "import onnxruntime as ort; \
                         from fastapi import FastAPI; import uvicorn; \
                         app=FastAPI(); session=ort.InferenceSession('{model}'); \
                         uvicorn.run(app, host='0.0.0.0', port={port})"
                    );
                    let status = tokio::process::Command::new("python3")
                        .args(["-c", &code])
                        .status()
                        .await?;
                    let _ = status;
                }

                other => {
                    println!("Unknown engine: {other}");
                    println!("Available: vllm, trt, onnx");
                }
            }
        }

        ServeCommands::List => {
            println!("Running Inference Servers");
            println!("{}", "=".repeat(60));

            // Check common inference ports
            for port in [8080, 8081, 8082, 8000, 5000] {
                let check = std::net::TcpStream::connect_timeout(
                    &format!("127.0.0.1:{port}").parse().expect("valid addr"),
                    std::time::Duration::from_millis(200),
                );
                if check.is_ok() {
                    println!("  :{port} — active");
                }
            }
        }

        ServeCommands::Stop { model } => {
            // Try to find the process serving this model
            println!("Stopping server for: {model}");
            let output = Command::new("pkill")
                .args(["-f", &format!("vllm.*{model}")])
                .output();
            match output {
                Ok(o) if o.status.success() => println!("Server stopped."),
                _ => println!("No server found for {model}. Try: zernel serve list"),
            }
        }

        ServeCommands::Logs { model } => {
            println!("Showing logs for model: {model}");
            println!("(inference log streaming coming in future release)");
            println!("For now: journalctl -u zernel-serve-{model} -f");
        }

        ServeCommands::Benchmark { url, qps, duration } => {
            println!("Load Testing: {url}");
            println!("  QPS: {qps}");
            println!("  Duration: {duration}s");
            println!();

            let code = format!(
                "import requests, time, concurrent.futures, statistics; \
                 url='{url}/v1/models'; latencies=[]; errors=0; \
                 start=time.time(); \
                 with concurrent.futures.ThreadPoolExecutor(max_workers={qps}) as ex: \
                     while time.time()-start < {duration}: \
                         futs=[ex.submit(requests.get, url) for _ in range({qps})]; \
                         for f in futs: \
                             try: \
                                 r=f.result(timeout=5); latencies.append(r.elapsed.total_seconds()*1000); \
                             except: errors+=1; \
                         time.sleep(1); \
                 if latencies: \
                     print(f'Requests:  {{len(latencies)}}'); \
                     print(f'Errors:    {{errors}}'); \
                     print(f'p50:       {{statistics.median(latencies):.1f}} ms'); \
                     print(f'p99:       {{sorted(latencies)[int(len(latencies)*0.99)]:.1f}} ms'); \
                     print(f'Mean:      {{statistics.mean(latencies):.1f}} ms'); \
                     print(f'Throughput: {{len(latencies)/{duration}:.0f}} req/s')"
            );
            let output = Command::new("python3").args(["-c", &code]).output()?;
            print!("{}", String::from_utf8_lossy(&output.stdout));
            if !output.status.success() {
                print!("{}", String::from_utf8_lossy(&output.stderr));
            }
        }
    }
    Ok(())
}
