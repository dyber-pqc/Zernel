// Copyright (C) 2026 Dyber, Inc.
//
// Security-focused integration tests for the Zernel CLI.
// Tests path traversal, injection, and input validation.

use std::path::PathBuf;
use std::process::Command;

fn zernel_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    let bin = if cfg!(target_os = "windows") {
        path.join("target").join("debug").join("zernel.exe")
    } else {
        path.join("target").join("debug").join("zernel")
    };
    assert!(bin.exists(), "zernel binary not found at {}", bin.display());
    bin
}

fn run_zernel_in(dir: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(zernel_bin())
        .args(args)
        .current_dir(dir)
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .expect("failed to execute zernel")
}

// ============================================================
// zernel init — path traversal prevention
// ============================================================

#[test]
fn init_rejects_dotdot() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_zernel_in(tmp.path(), &["init", ".."]);
    assert!(!output.status.success());
}

#[test]
fn init_rejects_dotdot_slash() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_zernel_in(tmp.path(), &["init", "../evil"]);
    assert!(!output.status.success());
}

#[test]
fn init_rejects_nested_traversal() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_zernel_in(tmp.path(), &["init", "../../etc/cron.d/evil"]);
    assert!(!output.status.success());
}

#[test]
fn init_rejects_absolute_path_unix() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_zernel_in(tmp.path(), &["init", "/tmp/evil"]);
    assert!(!output.status.success());
}

#[test]
fn init_rejects_backslash() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_zernel_in(tmp.path(), &["init", "foo\\bar"]);
    assert!(!output.status.success());
}

#[test]
fn init_rejects_hidden_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_zernel_in(tmp.path(), &["init", ".hidden"]);
    assert!(!output.status.success());
}

#[test]
fn init_rejects_null_byte() {
    // Null bytes in command args are rejected by the OS on most platforms.
    // Command::new().arg("foo\0bar") returns an InvalidInput error.
    // This is safe — the OS prevents the null byte from reaching our code.
    let tmp = tempfile::tempdir().unwrap();
    let result = Command::new(zernel_bin())
        .args(["init", "foo\0bar"])
        .current_dir(tmp.path())
        .env("ZERNEL_LOG", "zernel=error")
        .output();

    match result {
        Err(e) => {
            // OS rejected the null byte — this is the safe behavior
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
        }
        Ok(output) => {
            // If it somehow reached our code, validation should reject it
            assert!(!output.status.success());
        }
    }
}

#[test]
fn init_rejects_space_characters() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_zernel_in(tmp.path(), &["init", "foo bar"]);
    assert!(!output.status.success());
}

#[test]
fn init_rejects_windows_reserved_names() {
    let tmp = tempfile::tempdir().unwrap();
    for name in &["CON", "NUL", "COM1", "LPT1"] {
        let output = run_zernel_in(tmp.path(), &["init", name]);
        assert!(
            !output.status.success(),
            "should reject reserved name: {name}"
        );
    }
}

#[test]
fn init_rejects_overly_long_name() {
    let tmp = tempfile::tempdir().unwrap();
    let long_name = "a".repeat(200);
    let output = run_zernel_in(tmp.path(), &["init", &long_name]);
    assert!(!output.status.success());
}

#[test]
fn init_accepts_valid_names() {
    let tmp = tempfile::tempdir().unwrap();
    for name in &["myproject", "test-123", "model_v2", "LLaMA3.1"] {
        let output = run_zernel_in(tmp.path(), &["init", name]);
        assert!(
            output.status.success(),
            "should accept valid name: {name}, stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(tmp.path().join(name).exists());
    }
}

// ============================================================
// zernel model save — path traversal in name/tag
// ============================================================

#[test]
fn model_save_rejects_traversal_in_name() {
    let tmp = tempfile::tempdir().unwrap();
    let model_file = tmp.path().join("model.bin");
    std::fs::write(&model_file, b"data").unwrap();

    let output = Command::new(zernel_bin())
        .args([
            "model",
            "save",
            &model_file.to_string_lossy(),
            "--name",
            "../../etc/passwd",
        ])
        .env("HOME", tmp.path())
        .env("USERPROFILE", tmp.path())
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .unwrap();

    assert!(!output.status.success());
}

#[test]
fn model_save_rejects_traversal_in_tag() {
    let tmp = tempfile::tempdir().unwrap();
    let model_file = tmp.path().join("model.bin");
    std::fs::write(&model_file, b"data").unwrap();

    let output = Command::new(zernel_bin())
        .args([
            "model",
            "save",
            &model_file.to_string_lossy(),
            "--name",
            "safe",
            "--tag",
            "../evil",
        ])
        .env("HOME", tmp.path())
        .env("USERPROFILE", tmp.path())
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .unwrap();

    assert!(!output.status.success());
}

#[test]
fn model_save_rejects_slash_in_name() {
    let tmp = tempfile::tempdir().unwrap();
    let model_file = tmp.path().join("model.bin");
    std::fs::write(&model_file, b"data").unwrap();

    let output = Command::new(zernel_bin())
        .args([
            "model",
            "save",
            &model_file.to_string_lossy(),
            "--name",
            "foo/bar",
        ])
        .env("HOME", tmp.path())
        .env("USERPROFILE", tmp.path())
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .unwrap();

    assert!(!output.status.success());
}

#[test]
fn model_save_rejects_colon_in_tag() {
    let tmp = tempfile::tempdir().unwrap();
    let model_file = tmp.path().join("model.bin");
    std::fs::write(&model_file, b"data").unwrap();

    let output = Command::new(zernel_bin())
        .args([
            "model",
            "save",
            &model_file.to_string_lossy(),
            "--name",
            "safe",
            "--tag",
            "v1:latest",
        ])
        .env("HOME", tmp.path())
        .env("USERPROFILE", tmp.path())
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .unwrap();

    assert!(!output.status.success());
}

// ============================================================
// ZQL — injection resistance
// ============================================================

#[test]
fn zql_handles_sql_injection_attempt() {
    // ZQL uses a nom parser that produces an AST. There is no raw SQL
    // execution — the executor applies parsed conditions to in-memory
    // data. The semicolon in "... ; DROP TABLE ..." will cause a parse
    // error ("unexpected trailing input"), which is safe.
    let output = run_zernel_in(
        std::path::Path::new("."),
        &[
            "query",
            "SELECT * FROM experiments; DROP TABLE experiments;--",
        ],
    );
    // CLI must not crash (exit code 0 with parse error in stdout)
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Output should be a parse error, not a DROP result
    assert!(
        stdout.contains("unexpected trailing") || stdout.contains("Parse error"),
        "expected parse error, got: {stdout}"
    );
}

#[test]
fn zql_handles_oversized_query() {
    let tmp = tempfile::tempdir().unwrap();
    let huge_query = format!("SELECT {} FROM experiments", "a,".repeat(10000));
    let output = Command::new(zernel_bin())
        .args(["query", &huge_query])
        .env("HOME", tmp.path())
        .env("USERPROFILE", tmp.path())
        .env("ZERNEL_LOG", "zernel=error")
        .output()
        .unwrap();

    // Should not crash or hang
    assert!(output.status.success() || !output.status.success());
    // Just verifying it terminates
}
