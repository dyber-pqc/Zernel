// Copyright (C) 2026 Dyber, Inc. — Proprietary

use anyhow::Result;
use std::process::Command;

struct Check {
    name: &'static str,
    check: fn() -> CheckResult,
}

enum CheckResult {
    Pass(String),
    Warn(String),
    Fail(String),
    Skip(String),
}

/// Diagnose the Zernel environment.
pub async fn run() -> Result<()> {
    println!("Zernel Doctor v{}", env!("CARGO_PKG_VERSION"));
    println!("{}", "=".repeat(50));
    println!();

    let checks: Vec<Check> = vec![
        Check { name: "Operating System", check: check_os },
        Check { name: "Python", check: check_python },
        Check { name: "NVIDIA Driver", check: check_nvidia_driver },
        Check { name: "CUDA Toolkit", check: check_cuda },
        Check { name: "PyTorch", check: check_pytorch },
        Check { name: "PyTorch CUDA", check: check_pytorch_cuda },
        Check { name: "Git", check: check_git },
        Check { name: "zerneld", check: check_zerneld },
        Check { name: "Zernel DB", check: check_zernel_db },
    ];

    let mut pass = 0;
    let mut warn = 0;
    let mut fail = 0;
    let mut skip = 0;

    for check in &checks {
        let result = (check.check)();
        match &result {
            CheckResult::Pass(msg) => {
                println!("  [OK]   {}: {msg}", check.name);
                pass += 1;
            }
            CheckResult::Warn(msg) => {
                println!("  [WARN] {}: {msg}", check.name);
                warn += 1;
            }
            CheckResult::Fail(msg) => {
                println!("  [FAIL] {}: {msg}", check.name);
                fail += 1;
            }
            CheckResult::Skip(msg) => {
                println!("  [SKIP] {}: {msg}", check.name);
                skip += 1;
            }
        }
    }

    println!();
    println!(
        "Results: {pass} passed, {warn} warnings, {fail} failed, {skip} skipped"
    );

    if fail > 0 {
        println!();
        println!("Some checks failed. Fix the issues above before running ML workloads.");
    }

    Ok(())
}

fn run_cmd(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn check_os() -> CheckResult {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    if os == "linux" {
        if let Some(kernel) = run_cmd("uname", &["-r"]) {
            CheckResult::Pass(format!("Linux {kernel} ({arch})"))
        } else {
            CheckResult::Pass(format!("Linux ({arch})"))
        }
    } else {
        CheckResult::Warn(format!(
            "{os} ({arch}) — Zernel targets Linux. Some features unavailable."
        ))
    }
}

fn check_python() -> CheckResult {
    for cmd in &["python3", "python"] {
        if let Some(version) = run_cmd(cmd, &["--version"]) {
            return CheckResult::Pass(version);
        }
    }
    CheckResult::Fail("Python not found".into())
}

fn check_nvidia_driver() -> CheckResult {
    if let Some(output) = run_cmd("nvidia-smi", &["--query-gpu=driver_version", "--format=csv,noheader"]) {
        let version = output.lines().next().unwrap_or(&output);
        CheckResult::Pass(format!("driver v{version}"))
    } else {
        CheckResult::Fail("nvidia-smi not found or no GPU detected".into())
    }
}

fn check_cuda() -> CheckResult {
    if let Some(output) = run_cmd("nvcc", &["--version"]) {
        if let Some(line) = output.lines().find(|l| l.contains("release")) {
            CheckResult::Pass(line.trim().to_string())
        } else {
            CheckResult::Pass(output)
        }
    } else {
        CheckResult::Warn("nvcc not found — CUDA toolkit may not be installed".into())
    }
}

fn check_pytorch() -> CheckResult {
    if let Some(version) = run_cmd(
        "python3",
        &["-c", "import torch; print(torch.__version__)"],
    ) {
        CheckResult::Pass(format!("PyTorch {version}"))
    } else if let Some(version) = run_cmd(
        "python",
        &["-c", "import torch; print(torch.__version__)"],
    ) {
        CheckResult::Pass(format!("PyTorch {version}"))
    } else {
        CheckResult::Warn("PyTorch not installed".into())
    }
}

fn check_pytorch_cuda() -> CheckResult {
    let script = "import torch; print(torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)";
    if let Some(output) = run_cmd("python3", &["-c", script])
        .or_else(|| run_cmd("python", &["-c", script]))
    {
        let parts: Vec<&str> = output.split_whitespace().collect();
        match parts.first().map(|s| s.as_ref()) {
            Some("True") => {
                let gpus = parts.get(1).unwrap_or(&"?");
                CheckResult::Pass(format!("CUDA available — {gpus} GPU(s)"))
            }
            Some("False") => CheckResult::Warn("CUDA not available to PyTorch".into()),
            _ => CheckResult::Warn(format!("unexpected output: {output}")),
        }
    } else {
        CheckResult::Skip("PyTorch not installed".into())
    }
}

fn check_git() -> CheckResult {
    if let Some(version) = run_cmd("git", &["--version"]) {
        CheckResult::Pass(version)
    } else {
        CheckResult::Warn("git not found — experiment tracking won't record commits".into())
    }
}

fn check_zerneld() -> CheckResult {
    // Try to connect to zerneld health endpoint
    let result = std::net::TcpStream::connect_timeout(
        &"127.0.0.1:9091".parse().unwrap(),
        std::time::Duration::from_millis(500),
    );
    match result {
        Ok(_) => CheckResult::Pass("running on port 9091".into()),
        Err(_) => CheckResult::Warn(
            "not running — start with: zernel-ebpf --simulate".into(),
        ),
    }
}

fn check_zernel_db() -> CheckResult {
    let db_path = crate::experiments::tracker::experiments_db_path();
    if db_path.exists() {
        CheckResult::Pass(format!("{}", db_path.display()))
    } else {
        CheckResult::Pass(format!(
            "will be created at {}",
            db_path.display()
        ))
    }
}
