// Copyright (C) 2026 Dyber, Inc. — Proprietary

//! Kubernetes-based distributed training backend.
//!
//! Generates PyTorchJob YAML manifests and manages them via kubectl.

use anyhow::{Context, Result};
use std::path::Path;

/// Generate a Kubeflow PyTorchJob YAML manifest.
fn generate_pytorchjob_yaml(
    job_id: &str,
    image: &str,
    script: &str,
    gpus_per_node: u32,
    nodes: u32,
    namespace: &str,
    args: &[String],
) -> String {
    let script_args = if args.is_empty() {
        String::new()
    } else {
        format!(
            "\n{}",
            args.iter()
                .map(|a| format!("            - \"{}\"", a))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    format!(
        r#"apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {job_id}
  namespace: {namespace}
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: {image}
              command:
                - python3
                - {script}{script_args}
              resources:
                limits:
                  nvidia.com/gpu: {gpus_per_node}
              env:
                - name: NCCL_DEBUG
                  value: "WARN"
    Worker:
      replicas: {workers}
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: {image}
              command:
                - python3
                - {script}{script_args}
              resources:
                limits:
                  nvidia.com/gpu: {gpus_per_node}
              env:
                - name: NCCL_DEBUG
                  value: "WARN"
"#,
        job_id = job_id,
        namespace = namespace,
        image = image,
        script = script,
        gpus_per_node = gpus_per_node,
        workers = nodes.saturating_sub(1),
        script_args = script_args,
    )
}

/// Submit a distributed training job to Kubernetes via PyTorchJob CRD.
#[allow(clippy::too_many_arguments)]
pub async fn run_k8s_job(
    job_id: &str,
    script: &str,
    image: &str,
    gpus_per_node: u32,
    nodes: u32,
    namespace: &str,
    args: &[String],
    log_dir: &Path,
) -> Result<i32> {
    // Check kubectl
    let kubectl_check = std::process::Command::new("kubectl")
        .args(["version", "--client", "--short"])
        .output();

    match kubectl_check {
        Ok(o) if o.status.success() => {}
        _ => anyhow::bail!("kubectl not found. Install Kubernetes CLI first."),
    }

    // Generate manifest
    let yaml =
        generate_pytorchjob_yaml(job_id, image, script, gpus_per_node, nodes, namespace, args);
    let manifest_path = log_dir.join("pytorchjob.yaml");
    std::fs::write(&manifest_path, &yaml)?;

    println!("Kubernetes PyTorchJob");
    println!("  Image:     {image}");
    println!("  Nodes:     {nodes} (1 master + {} workers)", nodes - 1);
    println!("  GPUs/node: {gpus_per_node}");
    println!("  Namespace: {namespace}");
    println!("  Manifest:  {}", manifest_path.display());
    println!();

    // Apply manifest
    println!("Applying PyTorchJob...");
    let apply = tokio::process::Command::new("kubectl")
        .args(["apply", "-f"])
        .arg(&manifest_path)
        .status()
        .await
        .with_context(|| "kubectl apply failed")?;

    if !apply.success() {
        anyhow::bail!("kubectl apply failed for {}", manifest_path.display());
    }

    println!("Job submitted: {job_id}");
    println!();

    // Poll pod status
    println!("Waiting for pods...");
    let mut last_status = String::new();
    for _ in 0..120 {
        // 10 min timeout
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

        let status_output = tokio::process::Command::new("kubectl")
            .args([
                "get",
                "pods",
                "-l",
                &format!("training.kubeflow.org/job-name={job_id}"),
                "-n",
                namespace,
                "--no-headers",
                "-o",
                "custom-columns=NAME:.metadata.name,STATUS:.status.phase",
            ])
            .output()
            .await;

        if let Ok(output) = status_output {
            let status = String::from_utf8_lossy(&output.stdout).to_string();
            if status != last_status {
                print!("{status}");
                last_status = status.clone();
            }

            if status.contains("Running") || status.contains("Succeeded") {
                break;
            }
            if status.contains("Failed") || status.contains("Error") {
                println!("Job failed. Check: kubectl logs -l training.kubeflow.org/job-name={job_id} -n {namespace}");
                return Ok(1);
            }
        }
    }

    // Stream master logs
    println!();
    println!("--- Master logs ---");
    let log_status = tokio::process::Command::new("kubectl")
        .args([
            "logs",
            "-f",
            "-l",
            &format!(
                "training.kubeflow.org/job-name={job_id},training.kubeflow.org/replica-type=master"
            ),
            "-n",
            namespace,
        ])
        .status()
        .await
        .unwrap_or_default();

    Ok(if log_status.success() { 0 } else { 1 })
}

/// Cancel a Kubernetes job.
pub async fn cancel_k8s_job(job_id: &str, namespace: &str) -> Result<()> {
    println!("Deleting PyTorchJob {job_id}...");
    let status = tokio::process::Command::new("kubectl")
        .args(["delete", "pytorchjob", job_id, "-n", namespace])
        .status()
        .await?;

    if status.success() {
        println!("Job deleted.");
    } else {
        println!("Failed to delete job (may already be cleaned up).");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_valid_yaml() {
        let yaml = generate_pytorchjob_yaml(
            "test-job",
            "myimage:latest",
            "train.py",
            4,
            3,
            "default",
            &["--epochs".into(), "10".into()],
        );
        assert!(yaml.contains("kind: PyTorchJob"));
        assert!(yaml.contains("test-job"));
        assert!(yaml.contains("myimage:latest"));
        assert!(yaml.contains("nvidia.com/gpu: 4"));
        assert!(yaml.contains("replicas: 2")); // 3 nodes - 1 master = 2 workers
        assert!(yaml.contains("--epochs"));
    }
}
