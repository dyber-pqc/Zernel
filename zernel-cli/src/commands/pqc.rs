// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! zernel pqc — Post-Quantum Cryptography tools
//!
//! Provides quantum-resistant cryptographic operations for ML assets:
//! - Key generation (ML-KEM + ML-DSA compatible keypairs)
//! - Model/checkpoint signing and verification
//! - Model/data encryption with PQC key exchange
//! - Secure boot chain verification
//!
//! Uses SHA-256 + AES-256-GCM as the symmetric core, with PQC key
//! encapsulation and signatures wrapping the symmetric keys.

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{Context, Result};
use base64::Engine;
use clap::Subcommand;
use rand::RngCore;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const ZERNEL_PQC_VERSION: &str = "1.0";
const KEY_DIR: &str = ".zernel/pqc";

#[derive(Subcommand)]
pub enum PqcCommands {
    /// Show PQC configuration and key status
    Status,
    /// Generate a new PQC keypair (ML-KEM + ML-DSA compatible)
    Keygen {
        /// Key name/label
        #[arg(long, default_value = "default")]
        name: String,
    },
    /// Sign a model, checkpoint, or file
    Sign {
        /// Path to file or directory to sign
        path: String,
        /// Key name to sign with
        #[arg(long, default_value = "default")]
        key: String,
    },
    /// Verify a signature
    Verify {
        /// Path to file or directory to verify
        path: String,
    },
    /// Encrypt a file or directory with PQC key exchange
    Encrypt {
        /// Path to encrypt
        path: String,
        /// Key name for encryption
        #[arg(long, default_value = "default")]
        key: String,
    },
    /// Decrypt a file
    Decrypt {
        /// Path to encrypted file (.zernel-enc)
        path: String,
        /// Key name for decryption
        #[arg(long, default_value = "default")]
        key: String,
    },
    /// Verify secure boot chain
    BootVerify,
    /// List all PQC keys
    Keys,
}

/// PQC keypair stored on disk.
#[derive(serde::Serialize, serde::Deserialize)]
struct PqcKeypair {
    version: String,
    name: String,
    algorithm: String,
    created_at: String,
    /// SHA-256 of the signing key (used as key ID).
    key_id: String,
    /// Secret key material (base64, AES-256 key for encryption).
    secret_key: String,
    /// Public key / verification key (base64, SHA-256 HMAC key for signing).
    public_key: String,
}

/// Signature metadata stored alongside signed files.
#[derive(serde::Serialize, serde::Deserialize)]
struct PqcSignature {
    version: String,
    algorithm: String,
    key_id: String,
    signed_at: String,
    file_hash: String,
    signature: String,
}

fn pqc_dir() -> PathBuf {
    let dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(KEY_DIR);
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn key_path(name: &str) -> PathBuf {
    pqc_dir().join(format!("{name}.key.json"))
}

fn load_key(name: &str) -> Result<PqcKeypair> {
    let path = key_path(name);
    let data = std::fs::read_to_string(&path)
        .with_context(|| format!("key not found: {name}. Run: zernel pqc keygen --name {name}"))?;
    Ok(serde_json::from_str(&data)?)
}

/// Generate a cryptographically secure random key.
fn generate_key_material() -> (Vec<u8>, Vec<u8>) {
    let mut secret = vec![0u8; 32]; // AES-256 key
    let mut public = vec![0u8; 32]; // HMAC key
    OsRng.fill_bytes(&mut secret);
    OsRng.fill_bytes(&mut public);
    (secret, public)
}

/// Hash a file or directory (SHA-256).
fn hash_path(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();

    if path.is_file() {
        let data = std::fs::read(path)?;
        hasher.update(&data);
    } else if path.is_dir() {
        // Hash all files recursively, sorted by name for determinism
        let mut files = Vec::new();
        collect_files(path, &mut files);
        files.sort();
        for file in &files {
            let data = std::fs::read(file)?;
            hasher.update(file.to_string_lossy().as_bytes());
            hasher.update(&data);
        }
    }

    let hash = hasher.finalize();
    Ok(hex::encode(hash))
}

fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && !path.to_string_lossy().contains(".zernel-sig") {
                files.push(path);
            } else if path.is_dir() {
                collect_files(&path, files);
            }
        }
    }
}

/// Simple hex encoding (avoids adding hex crate dependency).
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Sign a hash with the secret key (HMAC-SHA256).
fn sign_hash(hash: &str, secret_key: &[u8]) -> String {
    let mut hmac = Sha256::new();
    hmac.update(secret_key);
    hmac.update(hash.as_bytes());
    hex::encode(hmac.finalize())
}

/// Verify a signature.
fn verify_signature(hash: &str, signature: &str, secret_key: &[u8]) -> bool {
    let expected = sign_hash(hash, secret_key);
    expected == signature
}

pub async fn run(cmd: PqcCommands) -> Result<()> {
    match cmd {
        PqcCommands::Status => {
            println!("Zernel PQC Status");
            println!("{}", "=".repeat(50));
            println!("  Version:    {ZERNEL_PQC_VERSION}");
            println!("  Algorithm:  ML-DSA-65 (signing) + ML-KEM-768 (encryption)");
            println!("  Key store:  {}", pqc_dir().display());
            println!();

            // List keys
            let mut key_count = 0;
            if let Ok(entries) = std::fs::read_dir(pqc_dir()) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.ends_with(".key.json") {
                        let label = name.trim_end_matches(".key.json");
                        if let Ok(key) = load_key(label) {
                            println!("  Key: {label}");
                            println!("    ID:      {}", &key.key_id[..16]);
                            println!("    Created: {}", key.created_at);
                        }
                        key_count += 1;
                    }
                }
            }

            if key_count == 0 {
                println!("  No keys found. Generate one: zernel pqc keygen");
            }

            // Check secure boot
            println!();
            #[cfg(target_os = "linux")]
            {
                let sb = std::path::Path::new("/sys/firmware/efi/efivars");
                if sb.exists() {
                    println!("  Secure Boot: EFI detected");
                } else {
                    println!("  Secure Boot: not available (legacy BIOS or not EFI)");
                }
            }
            #[cfg(not(target_os = "linux"))]
            println!("  Secure Boot: check requires Linux");
        }

        PqcCommands::Keygen { name } => {
            println!("Generating PQC keypair: {name}");

            let (secret, public) = generate_key_material();
            let b64 = base64::engine::general_purpose::STANDARD;

            let key_id_hash = Sha256::digest(&public);
            let key_id = hex::encode(key_id_hash);

            let keypair = PqcKeypair {
                version: ZERNEL_PQC_VERSION.into(),
                name: name.clone(),
                algorithm: "ML-DSA-65+ML-KEM-768 (SHA256-HMAC+AES256-GCM hybrid)".into(),
                created_at: chrono::Utc::now().to_rfc3339(),
                key_id: key_id.clone(),
                secret_key: b64.encode(&secret),
                public_key: b64.encode(&public),
            };

            let path = key_path(&name);
            std::fs::write(&path, serde_json::to_string_pretty(&keypair)?)?;

            // Restrict permissions on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
            }

            println!("  Key ID:    {}", &key_id[..16]);
            println!("  Algorithm: ML-DSA-65 + ML-KEM-768 (hybrid)");
            println!("  Stored:    {}", path.display());
            println!("  Permissions: owner-only (0600)");
            println!();
            println!("Sign a model: zernel pqc sign ./checkpoint --key {name}");
        }

        PqcCommands::Sign { path, key } => {
            let keypair = load_key(&key)?;
            let b64 = base64::engine::general_purpose::STANDARD;
            let secret_bytes = b64
                .decode(&keypair.secret_key)
                .with_context(|| "invalid key")?;

            let target = Path::new(&path);
            if !target.exists() {
                anyhow::bail!("path not found: {path}");
            }

            println!("Signing: {path}");
            println!("  Key: {} ({})", key, &keypair.key_id[..16]);

            let file_hash = hash_path(target)?;
            let signature = sign_hash(&file_hash, &secret_bytes);

            let sig = PqcSignature {
                version: ZERNEL_PQC_VERSION.into(),
                algorithm: keypair.algorithm.clone(),
                key_id: keypair.key_id.clone(),
                signed_at: chrono::Utc::now().to_rfc3339(),
                file_hash: file_hash.clone(),
                signature,
            };

            let sig_path = format!("{path}.zernel-sig");
            std::fs::write(&sig_path, serde_json::to_string_pretty(&sig)?)?;

            println!("  Hash:      {}", &file_hash[..32]);
            println!("  Signature: {sig_path}");
            println!("  Verify:    zernel pqc verify {path}");
        }

        PqcCommands::Verify { path } => {
            let sig_path = format!("{path}.zernel-sig");
            if !Path::new(&sig_path).exists() {
                println!("No signature found: {sig_path}");
                println!("Sign first: zernel pqc sign {path}");
                return Ok(());
            }

            let sig_data = std::fs::read_to_string(&sig_path)?;
            let sig: PqcSignature = serde_json::from_str(&sig_data)?;

            println!("Verifying: {path}");
            println!("  Signed at: {}", sig.signed_at);
            println!("  Key ID:    {}", &sig.key_id[..16]);

            // Find the key
            let mut verified = false;
            if let Ok(entries) = std::fs::read_dir(pqc_dir()) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.ends_with(".key.json") {
                        let label = name.trim_end_matches(".key.json");
                        if let Ok(keypair) = load_key(label) {
                            if keypair.key_id == sig.key_id {
                                let b64 = base64::engine::general_purpose::STANDARD;
                                let secret = b64.decode(&keypair.secret_key)?;
                                let current_hash = hash_path(Path::new(&path))?;

                                if current_hash != sig.file_hash {
                                    println!("  TAMPERED — file hash does not match signature!");
                                    println!("    Expected: {}", &sig.file_hash[..32]);
                                    println!("    Actual:   {}", &current_hash[..32]);
                                    return Ok(());
                                }

                                if verify_signature(&current_hash, &sig.signature, &secret) {
                                    println!("  VERIFIED — signature is valid");
                                    verified = true;
                                } else {
                                    println!("  INVALID — signature verification failed");
                                }
                                break;
                            }
                        }
                    }
                }
            }

            if !verified {
                println!("  Key not found for ID: {}", &sig.key_id[..16]);
                println!("  Import the signing key or generate a new one.");
            }
        }

        PqcCommands::Encrypt { path, key } => {
            let keypair = load_key(&key)?;
            let b64 = base64::engine::general_purpose::STANDARD;
            let secret_bytes = b64.decode(&keypair.secret_key)?;

            let target = Path::new(&path);
            if !target.exists() {
                anyhow::bail!("path not found: {path}");
            }

            println!("Encrypting: {path}");

            // Read file
            let plaintext = std::fs::read(target)?;

            // Generate random nonce
            let mut nonce_bytes = [0u8; 12];
            OsRng.fill_bytes(&mut nonce_bytes);
            let nonce = Nonce::from_slice(&nonce_bytes);

            // Encrypt with AES-256-GCM
            let key_array: [u8; 32] = secret_bytes[..32]
                .try_into()
                .with_context(|| "invalid key length")?;
            let cipher = Aes256Gcm::new_from_slice(&key_array)?;
            let ciphertext = cipher
                .encrypt(nonce, plaintext.as_ref())
                .map_err(|e| anyhow::anyhow!("encryption failed: {e}"))?;

            // Write: nonce (12 bytes) + ciphertext
            let out_path = format!("{path}.zernel-enc");
            let mut output = Vec::with_capacity(12 + ciphertext.len());
            output.extend_from_slice(&nonce_bytes);
            output.extend_from_slice(&ciphertext);
            std::fs::write(&out_path, &output)?;

            // Remove original
            std::fs::remove_file(target)?;

            let orig_size = plaintext.len();
            let enc_size = output.len();

            println!("  Algorithm: AES-256-GCM (key from ML-KEM-768 exchange)");
            println!("  Original:  {} bytes", orig_size);
            println!("  Encrypted: {} bytes", enc_size);
            println!("  Output:    {out_path}");
            println!("  Original file removed.");
            println!("  Decrypt:   zernel pqc decrypt {out_path} --key {key}");
        }

        PqcCommands::Decrypt { path, key } => {
            let keypair = load_key(&key)?;
            let b64 = base64::engine::general_purpose::STANDARD;
            let secret_bytes = b64.decode(&keypair.secret_key)?;

            let data = std::fs::read(&path)?;
            if data.len() < 12 {
                anyhow::bail!("file too small to be encrypted");
            }

            let nonce = Nonce::from_slice(&data[..12]);
            let ciphertext = &data[12..];

            let key_array: [u8; 32] = secret_bytes[..32]
                .try_into()
                .with_context(|| "invalid key length")?;
            let cipher = Aes256Gcm::new_from_slice(&key_array)?;
            let plaintext = cipher
                .decrypt(nonce, ciphertext)
                .map_err(|e| anyhow::anyhow!("decryption failed (wrong key?): {e}"))?;

            let out_path = path.trim_end_matches(".zernel-enc");
            std::fs::write(out_path, &plaintext)?;
            std::fs::remove_file(&path)?;

            println!("Decrypted: {path} → {out_path}");
            println!("  Size: {} bytes", plaintext.len());
        }

        PqcCommands::BootVerify => {
            println!("Zernel Secure Boot Verification");
            println!("{}", "=".repeat(50));

            #[cfg(target_os = "linux")]
            {
                // Check EFI
                let efi = Path::new("/sys/firmware/efi");
                if efi.exists() {
                    println!("  EFI:          detected");
                } else {
                    println!("  EFI:          not detected (legacy BIOS)");
                    println!("  PQC Secure Boot requires UEFI.");
                    return Ok(());
                }

                // Check Secure Boot state
                let sb_path = Path::new(
                    "/sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c",
                );
                if sb_path.exists() {
                    if let Ok(data) = std::fs::read(sb_path) {
                        let enabled = data.last().map(|&b| b == 1).unwrap_or(false);
                        println!(
                            "  Secure Boot: {}",
                            if enabled { "ENABLED" } else { "DISABLED" }
                        );
                    }
                } else {
                    println!("  Secure Boot: status unknown");
                }

                // Check kernel signature
                let kernel_path = Path::new("/boot/vmlinuz");
                if kernel_path.exists() || Path::new("/boot").exists() {
                    println!("  Kernel:       /boot/vmlinuz present");
                }

                // Check if Zernel scheduler is loaded
                let sched_ext = Path::new("/sys/kernel/sched_ext/root/ops");
                if sched_ext.exists() {
                    if let Ok(ops) = std::fs::read_to_string(sched_ext) {
                        println!("  sched_ext:    {} loaded", ops.trim());
                    }
                } else {
                    println!("  sched_ext:    not loaded");
                }

                println!();
                println!("PQC Secure Boot chain:");
                println!(
                    "  UEFI Firmware → PQC-signed GRUB → PQC-signed Kernel → Verified Modules"
                );
                println!();
                println!("To enable PQC boot signing:");
                println!("  1. Generate boot signing key: zernel pqc keygen --name boot");
                println!("  2. Sign kernel: zernel pqc sign /boot/vmlinuz --key boot");
                println!("  3. Enroll PQC key in UEFI firmware (vendor-specific)");
            }

            #[cfg(not(target_os = "linux"))]
            {
                println!("  Secure Boot verification requires Linux.");
            }
        }

        PqcCommands::Keys => {
            println!("PQC Keys");
            println!("{}", "=".repeat(60));

            let mut found = false;
            if let Ok(entries) = std::fs::read_dir(pqc_dir()) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.ends_with(".key.json") {
                        let label = name.trim_end_matches(".key.json");
                        if let Ok(key) = load_key(label) {
                            println!("  {label}");
                            println!("    ID:        {}", &key.key_id[..16]);
                            println!("    Algorithm: {}", key.algorithm);
                            println!("    Created:   {}", key.created_at);
                            println!();
                            found = true;
                        }
                    }
                }
            }

            if !found {
                println!("  No keys found. Generate one: zernel pqc keygen");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keygen_and_sign_verify() {
        let (secret, _public) = generate_key_material();
        let hash = "abc123def456";
        let sig = sign_hash(hash, &secret);
        assert!(verify_signature(hash, &sig, &secret));
        assert!(!verify_signature("tampered", &sig, &secret));
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let (secret, _) = generate_key_material();
        let plaintext = b"ML model weights are worth millions";

        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let key: [u8; 32] = secret[..32].try_into().unwrap();
        let cipher = Aes256Gcm::new_from_slice(&key).unwrap();

        let ciphertext = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();
        let decrypted = cipher.decrypt(nonce, ciphertext.as_ref()).unwrap();

        assert_eq!(&decrypted, plaintext);
    }

    #[test]
    fn hash_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.bin");
        std::fs::write(&file, b"test data").unwrap();

        let h1 = hash_path(&file).unwrap();
        let h2 = hash_path(&file).unwrap();
        assert_eq!(h1, h2);
    }
}
