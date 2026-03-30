// Copyright (C) 2026 Dyber, Inc.
//
// Integration tests for the Zernel CLI workflow.
// These tests invoke the `zernel` binary as a subprocess.

use std::path::PathBuf;
use std::process::Command;

fn zernel_bin() -> PathBuf {
    // Built by `cargo test --workspace` into workspace target/debug/
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> workspace root
    let bin = if cfg!(target_os = "windows") {
        path.join("target").join("debug").join("zernel.exe")
    } else {
        path.join("target").join("debug").join("zernel")
    };
    assert!(bin.exists(), "zernel binary not found at {}", bin.display());
    bin
}

fn run_zernel(args: &[&str]) -> std::process::Output {
    Command::new(zernel_bin())
        .args(args)
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .expect("failed to execute zernel")
}

fn run_zernel_in(dir: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(zernel_bin())
        .args(args)
        .current_dir(dir)
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .expect("failed to execute zernel")
}

#[test]
fn zernel_version() {
    let output = run_zernel(&["--version"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("zernel"));
}

#[test]
fn zernel_help() {
    let output = run_zernel(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("AI-Native"));
    assert!(stdout.contains("init"));
    assert!(stdout.contains("run"));
    assert!(stdout.contains("watch"));
    assert!(stdout.contains("exp"));
    assert!(stdout.contains("model"));
    assert!(stdout.contains("job"));
    assert!(stdout.contains("doctor"));
    assert!(stdout.contains("query"));
}

#[test]
fn zernel_init_creates_project() {
    let tmp = tempfile::tempdir().unwrap();
    let project_name = "test-project";

    let output = run_zernel_in(tmp.path(), &["init", project_name]);
    assert!(
        output.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let project_dir = tmp.path().join(project_name);
    assert!(project_dir.exists());
    assert!(project_dir.join("zernel.toml").exists());
    assert!(project_dir.join("train.py").exists());
    assert!(project_dir.join("data").exists());
    assert!(project_dir.join("models").exists());
    assert!(project_dir.join("configs").exists());

    // Verify zernel.toml content
    let config = std::fs::read_to_string(project_dir.join("zernel.toml")).unwrap();
    assert!(config.contains(project_name));
    assert!(config.contains("pytorch"));
}

#[test]
fn zernel_init_rejects_path_traversal() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_zernel_in(tmp.path(), &["init", "../escape"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stderr}{stdout}");
    assert!(
        combined.contains("path separator") || combined.contains(".."),
        "expected path traversal error, got: {combined}"
    );
}

#[test]
fn zernel_init_rejects_absolute_path() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_zernel_in(tmp.path(), &["init", "/tmp/evil"]);
    assert!(!output.status.success());
}

#[test]
fn zernel_exp_list_runs() {
    // Just verify the command runs without crashing.
    // In a fresh HOME it shows "No experiments", but in a real HOME
    // it may show existing experiments — both are valid.
    let output = run_zernel(&["exp", "list"]);
    assert!(output.status.success());
}

#[test]
fn zernel_doctor_runs() {
    let output = run_zernel(&["doctor"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Zernel Doctor"));
    assert!(stdout.contains("Operating System"));
    assert!(stdout.contains("Python"));
}

#[test]
fn zernel_query_runs() {
    let output = run_zernel(&["query", "SELECT * FROM experiments"]);
    assert!(output.status.success());
}

#[test]
fn zernel_query_invalid_table() {
    let tmp = tempfile::tempdir().unwrap();
    let output = Command::new(zernel_bin())
        .args(["query", "SELECT * FROM nonexistent"])
        .env("HOME", tmp.path())
        .env("USERPROFILE", tmp.path())
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Unknown table"));
}

#[test]
fn zernel_query_parse_error() {
    let output = run_zernel(&["query", "not a valid query"]);
    assert!(output.status.success()); // parse error is not a CLI failure
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Parse error") || stdout.contains("parse error"));
}

#[test]
fn zernel_model_list_runs() {
    let output = run_zernel(&["model", "list"]);
    assert!(output.status.success());
}

#[test]
fn zernel_job_list_runs() {
    let output = run_zernel(&["job", "list"]);
    assert!(output.status.success());
}

#[test]
fn zernel_run_with_demo_script() {
    let tmp = tempfile::tempdir().unwrap();

    // Create a tiny Python script that outputs metrics
    let script = tmp.path().join("demo.py");
    std::fs::write(
        &script,
        "print('step: 1 loss: 0.5 accuracy: 0.8')\nprint('step: 2 loss: 0.3 accuracy: 0.9')\n",
    )
    .unwrap();

    let output = Command::new(zernel_bin())
        .args(["run", "demo.py"])
        .current_dir(tmp.path())
        .env("HOME", tmp.path())
        .env("USERPROFILE", tmp.path())
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Script may fail if python3 not found, which is OK
    if output.status.success() {
        assert!(stdout.contains("Zernel Run"));
        assert!(stdout.contains("loss"));
        assert!(stdout.contains("Experiment:") || stdout.contains("exp-"));
    } else {
        // Acceptable: python not installed in CI
        assert!(
            stderr.contains("python") || stderr.contains("failed to launch"),
            "unexpected error: {stderr}"
        );
    }
}

#[test]
fn zernel_model_save_and_list() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("checkpoint");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("model.bin"), b"fake model data").unwrap();

    // Save
    let output = Command::new(zernel_bin())
        .args([
            "model",
            "save",
            &model_dir.to_string_lossy(),
            "--name",
            "testmodel",
            "--tag",
            "v1",
        ])
        .env("HOME", tmp.path())
        .env("USERPROFILE", tmp.path())
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "save failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved model: testmodel:v1"));

    // List
    let output = Command::new(zernel_bin())
        .args(["model", "list"])
        .env("HOME", tmp.path())
        .env("USERPROFILE", tmp.path())
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("testmodel"));
}
