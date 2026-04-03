# Copyright (C) 2026 Dyber, Inc. — Proprietary
#
# Zernel ML Runtime — automatic training optimization
#
# Usage:
#   import zernel_runtime   # Call before importing torch
#   import torch             # PyTorch is now auto-optimized
#
# Or run any script with:
#   zernel-run train.py      # Automatically applies all optimizations
#
# What it does (zero code changes required):
#   1. Enables TF32 for matmul + cuDNN (Ampere+ GPUs, ~3x faster FP32)
#   2. Configures CUDA memory allocator (expandable_segments, reduces OOM)
#   3. Auto-wraps model.forward() with mixed precision (BF16/FP16)
#   4. Auto-applies torch.compile() to models (30-50% faster)
#   5. Reports energy savings estimates at end of training

from zernel_runtime.optimizer import auto_optimize

__version__ = "0.1.0"

# Auto-optimize on import
auto_optimize()
