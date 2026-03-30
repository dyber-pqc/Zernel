// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! Input validation for user-supplied names, tags, and paths.
//!
//! Prevents path traversal attacks and ensures safe filesystem operations.

use anyhow::{bail, Result};

/// Maximum length for project/model names.
const MAX_NAME_LEN: usize = 128;

/// Maximum length for tags.
const MAX_TAG_LEN: usize = 64;

/// Validate a project or model name.
///
/// Rejects names containing path separators, parent directory references,
/// null bytes, or other characters that could escape the intended directory.
///
/// Valid pattern: `^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$`
pub fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("name cannot be empty");
    }
    if name.len() > MAX_NAME_LEN {
        bail!("name exceeds maximum length of {MAX_NAME_LEN} characters");
    }
    if name.contains('\0') {
        bail!("name contains null byte");
    }
    if name.contains('/') || name.contains('\\') {
        bail!("name cannot contain path separators ('/' or '\\')");
    }
    if name.contains("..") {
        bail!("name cannot contain '..'");
    }
    if name.starts_with('.') {
        bail!("name cannot start with '.'");
    }
    if name.starts_with('-') {
        bail!("name cannot start with '-'");
    }

    // Must start with alphanumeric
    let first = name.chars().next().unwrap();
    if !first.is_ascii_alphanumeric() {
        bail!("name must start with a letter or digit");
    }

    // Only allow safe characters
    for ch in name.chars() {
        if !ch.is_ascii_alphanumeric() && ch != '.' && ch != '_' && ch != '-' {
            bail!("name contains invalid character: '{ch}'. Only letters, digits, '.', '_', '-' are allowed");
        }
    }

    // Reject Windows reserved names
    let upper = name.to_uppercase();
    let reserved = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
        "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    let stem = upper.split('.').next().unwrap_or(&upper);
    if reserved.contains(&stem) {
        bail!("name uses a reserved system name: {name}");
    }

    Ok(())
}

/// Validate a model tag.
///
/// Same rules as `validate_name` plus rejects ':' (used as name:tag separator)
/// and has a shorter maximum length.
pub fn validate_tag(tag: &str) -> Result<()> {
    if tag.is_empty() {
        bail!("tag cannot be empty");
    }
    if tag.len() > MAX_TAG_LEN {
        bail!("tag exceeds maximum length of {MAX_TAG_LEN} characters");
    }
    if tag.contains(':') {
        bail!("tag cannot contain ':'");
    }
    // Delegate the rest to validate_name (same character rules)
    validate_name(tag)
}

/// Validate that a path does not escape the current working directory.
///
/// Used for `zernel init` to prevent creating projects in arbitrary locations.
pub fn validate_project_path(name: &str) -> Result<()> {
    validate_name(name)?;

    // Additional check: the resolved path must not be absolute
    let path = std::path::Path::new(name);
    if path.is_absolute() {
        bail!("project name cannot be an absolute path");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_names() {
        assert!(validate_name("my-project").is_ok());
        assert!(validate_name("model_v2").is_ok());
        assert!(validate_name("llama3.1").is_ok());
        assert!(validate_name("A").is_ok());
        assert!(validate_name("test123").is_ok());
    }

    #[test]
    fn rejects_empty() {
        assert!(validate_name("").is_err());
    }

    #[test]
    fn rejects_path_traversal() {
        assert!(validate_name("../evil").is_err());
        assert!(validate_name("../../etc/passwd").is_err());
        assert!(validate_name("foo/../bar").is_err());
        assert!(validate_name("..").is_err());
    }

    #[test]
    fn rejects_path_separators() {
        assert!(validate_name("foo/bar").is_err());
        assert!(validate_name("foo\\bar").is_err());
        assert!(validate_name("/etc/passwd").is_err());
    }

    #[test]
    fn rejects_hidden_names() {
        assert!(validate_name(".hidden").is_err());
        assert!(validate_name(".git").is_err());
    }

    #[test]
    fn rejects_null_bytes() {
        assert!(validate_name("foo\0bar").is_err());
    }

    #[test]
    fn rejects_too_long() {
        let long = "a".repeat(129);
        assert!(validate_name(&long).is_err());
        let ok = "a".repeat(128);
        assert!(validate_name(&ok).is_ok());
    }

    #[test]
    fn rejects_windows_reserved() {
        assert!(validate_name("CON").is_err());
        assert!(validate_name("NUL").is_err());
        assert!(validate_name("COM1").is_err());
        assert!(validate_name("LPT1.txt").is_err());
    }

    #[test]
    fn rejects_special_chars() {
        assert!(validate_name("foo bar").is_err());
        assert!(validate_name("foo@bar").is_err());
        assert!(validate_name("foo$bar").is_err());
    }

    #[test]
    fn rejects_leading_dash() {
        assert!(validate_name("-flag").is_err());
    }

    #[test]
    fn tag_rejects_colon() {
        assert!(validate_tag("v1:latest").is_err());
        assert!(validate_tag("production").is_ok());
    }

    #[test]
    fn tag_max_length() {
        let long = "a".repeat(65);
        assert!(validate_tag(&long).is_err());
        let ok = "a".repeat(64);
        assert!(validate_tag(&ok).is_ok());
    }

    #[test]
    fn project_path_rejects_absolute() {
        assert!(validate_project_path("/tmp/evil").is_err());
    }
}
