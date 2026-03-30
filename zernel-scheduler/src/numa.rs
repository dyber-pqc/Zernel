// Copyright (C) 2026 Dyber, Inc. — GPL-2.0

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// NUMA topology information for a system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
    /// GPU-to-NUMA-node mapping (gpu_id -> node_id).
    pub gpu_node_map: HashMap<u32, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaNode {
    pub node_id: u32,
    pub cpu_ids: Vec<u32>,
    pub memory_mb: u64,
}

impl NumaTopology {
    /// Detect NUMA topology from the system.
    /// On non-Linux or single-node systems, returns a single flat node.
    pub fn detect() -> Self {
        // Try to read from /sys/devices/system/node/ on Linux
        #[cfg(target_os = "linux")]
        if let Ok(topo) = Self::detect_linux() {
            return topo;
        }

        // Fallback: single node with all available CPUs
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        Self {
            nodes: vec![NumaNode {
                node_id: 0,
                cpu_ids: (0..num_cpus).collect(),
                memory_mb: 0, // unknown
            }],
            gpu_node_map: HashMap::new(),
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> anyhow::Result<Self> {
        use std::fs;

        let mut nodes = Vec::new();
        let node_base = std::path::Path::new("/sys/devices/system/node");

        if !node_base.exists() {
            anyhow::bail!("no NUMA sysfs");
        }

        for entry in fs::read_dir(node_base)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("node") {
                continue;
            }
            let node_id: u32 = name.trim_start_matches("node").parse().unwrap_or(0);

            // Read CPU list
            let cpulist_path = entry.path().join("cpulist");
            let cpu_ids = if cpulist_path.exists() {
                parse_cpu_list(&fs::read_to_string(cpulist_path).unwrap_or_default())
            } else {
                vec![]
            };

            // Read memory info
            let meminfo_path = entry.path().join("meminfo");
            let memory_mb = if meminfo_path.exists() {
                parse_node_meminfo(&fs::read_to_string(meminfo_path).unwrap_or_default())
            } else {
                0
            };

            nodes.push(NumaNode {
                node_id,
                cpu_ids,
                memory_mb,
            });
        }

        nodes.sort_by_key(|n| n.node_id);

        // Detect GPU NUMA affinity via nvidia-smi or sysfs
        let gpu_node_map = detect_gpu_numa_map();

        Ok(Self {
            nodes,
            gpu_node_map,
        })
    }

    /// Get the NUMA node that a GPU is connected to.
    pub fn gpu_numa_node(&self, gpu_id: u32) -> Option<u32> {
        self.gpu_node_map.get(&gpu_id).copied()
    }

    /// Get CPUs on the same NUMA node as a given GPU.
    pub fn cpus_for_gpu(&self, gpu_id: u32) -> Vec<u32> {
        if let Some(node_id) = self.gpu_numa_node(gpu_id) {
            self.nodes
                .iter()
                .find(|n| n.node_id == node_id)
                .map(|n| n.cpu_ids.clone())
                .unwrap_or_default()
        } else {
            // No NUMA info — return all CPUs
            self.nodes
                .iter()
                .flat_map(|n| n.cpu_ids.iter().copied())
                .collect()
        }
    }

    /// Select the best CPU for a task given its GPU affinity.
    /// Returns the CPU ID from the preferred NUMA node with the lowest load.
    pub fn select_cpu(&self, gpu_id: Option<u32>, cpu_loads: &HashMap<u32, f32>) -> u32 {
        let preferred_cpus = match gpu_id {
            Some(gid) => self.cpus_for_gpu(gid),
            None => self
                .nodes
                .iter()
                .flat_map(|n| n.cpu_ids.iter().copied())
                .collect(),
        };

        // Pick the CPU with lowest load from preferred set
        preferred_cpus
            .iter()
            .copied()
            .min_by(|a, b| {
                let la = cpu_loads.get(a).unwrap_or(&0.0);
                let lb = cpu_loads.get(b).unwrap_or(&0.0);
                la.partial_cmp(lb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }

    pub fn total_cpus(&self) -> usize {
        self.nodes.iter().map(|n| n.cpu_ids.len()).sum()
    }
}

/// Parse a Linux CPU list string like "0-3,8-11" into individual CPU IDs.
fn parse_cpu_list(s: &str) -> Vec<u32> {
    let mut cpus = Vec::new();
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<u32>(), end.parse::<u32>()) {
                cpus.extend(s..=e);
            }
        } else if let Ok(cpu) = part.parse::<u32>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Parse MemTotal from a NUMA node meminfo file.
fn parse_node_meminfo(s: &str) -> u64 {
    for line in s.lines() {
        if line.contains("MemTotal") {
            // Format: "Node 0 MemTotal:       12345678 kB"
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                if let Ok(kb) = parts[3].parse::<u64>() {
                    return kb / 1024; // Convert kB to MB
                }
            }
        }
    }
    0
}

/// Detect which NUMA node each GPU is on.
#[cfg(target_os = "linux")]
fn detect_gpu_numa_map() -> HashMap<u32, u32> {
    let mut map = HashMap::new();
    // Try reading from /sys/bus/pci/devices/*/numa_node for NVIDIA GPUs
    // Each GPU PCI device has a numa_node file
    if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
        let mut gpu_idx = 0u32;
        for entry in entries.flatten() {
            let vendor_path = entry.path().join("vendor");
            if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                // NVIDIA vendor ID = 0x10de
                if vendor.trim() == "0x10de" {
                    let class_path = entry.path().join("class");
                    if let Ok(class) = std::fs::read_to_string(&class_path) {
                        // GPU class = 0x030000 or 0x030200
                        if class.trim().starts_with("0x0302") || class.trim().starts_with("0x0300")
                        {
                            let numa_path = entry.path().join("numa_node");
                            if let Ok(numa) = std::fs::read_to_string(&numa_path) {
                                if let Ok(node) = numa.trim().parse::<i32>() {
                                    if node >= 0 {
                                        map.insert(gpu_idx, node as u32);
                                    }
                                }
                            }
                            gpu_idx += 1;
                        }
                    }
                }
            }
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_list_simple() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn parse_cpu_list_ranges_and_singles() {
        assert_eq!(parse_cpu_list("0-2,5,8-9"), vec![0, 1, 2, 5, 8, 9]);
    }

    #[test]
    fn parse_cpu_list_empty() {
        assert_eq!(parse_cpu_list(""), Vec::<u32>::new());
    }

    #[test]
    fn single_node_topology() {
        let topo = NumaTopology::detect();
        assert!(!topo.nodes.is_empty());
        assert!(topo.total_cpus() > 0);
    }

    #[test]
    fn select_cpu_picks_lowest_load() {
        let topo = NumaTopology {
            nodes: vec![NumaNode {
                node_id: 0,
                cpu_ids: vec![0, 1, 2, 3],
                memory_mb: 32768,
            }],
            gpu_node_map: HashMap::from([(0, 0)]),
        };
        let loads = HashMap::from([(0, 0.9), (1, 0.2), (2, 0.5), (3, 0.8)]);
        assert_eq!(topo.select_cpu(Some(0), &loads), 1);
    }

    #[test]
    fn cpus_for_gpu_with_mapping() {
        let topo = NumaTopology {
            nodes: vec![
                NumaNode {
                    node_id: 0,
                    cpu_ids: vec![0, 1, 2, 3],
                    memory_mb: 16384,
                },
                NumaNode {
                    node_id: 1,
                    cpu_ids: vec![4, 5, 6, 7],
                    memory_mb: 16384,
                },
            ],
            gpu_node_map: HashMap::from([(0, 0), (1, 1)]),
        };
        assert_eq!(topo.cpus_for_gpu(0), vec![0, 1, 2, 3]);
        assert_eq!(topo.cpus_for_gpu(1), vec![4, 5, 6, 7]);
    }
}
