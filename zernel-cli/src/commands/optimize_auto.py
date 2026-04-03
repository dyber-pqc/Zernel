import torch, os, sys

script_path = os.environ.get('ZERNEL_OPT_SCRIPT', '')
output_path = os.environ.get('ZERNEL_OPT_OUTPUT', 'train_optimized.py')

if not os.path.exists(script_path):
    print(f"ERROR: Script not found: {script_path}")
    sys.exit(1)

with open(script_path) as f:
    original = f.read()

cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
cpu_count = os.cpu_count() or 4

optimizations = []
notes = []

header = f"#!/usr/bin/env python3\n# Auto-optimized by Zernel — https://zernel.org\n# Original: {script_path}\n#\n# Optimizations applied:\n"

# 1. TF32
if cap[0] >= 8:
    header += "#   - TF32 enabled (Ampere+ GPU detected)\n"
    optimizations.append("import torch")
    optimizations.append("torch.backends.cuda.matmul.allow_tf32 = True")
    optimizations.append("torch.backends.cudnn.allow_tf32 = True")
    print("  [+] TF32 enabled for matmul + cuDNN")

# 2. CUDA allocator
header += "#   - CUDA memory allocator optimized\n"
optimizations.append("import os")
optimizations.append("os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:512')")
print("  [+] CUDA allocator: expandable_segments + GC threshold")

# 3. Mixed precision
has_autocast = "autocast" in original
if not has_autocast:
    if cap[0] >= 8:
        header += "#   - BF16 mixed precision recommended\n"
        notes.append("# Wrap your forward pass with:\n#   with torch.autocast('cuda', dtype=torch.bfloat16):\n#       output = model(input)\n#       loss = loss_fn(output, target)\n#   loss.backward()")
        print("  [+] BF16 mixed precision recommendation added")
    elif cap[0] >= 7:
        header += "#   - FP16 mixed precision recommended\n"
        notes.append("# Use GradScaler for FP16:\n#   scaler = torch.cuda.amp.GradScaler()\n#   with torch.autocast('cuda', dtype=torch.float16):\n#       output = model(input); loss = loss_fn(output, target)\n#   scaler.scale(loss).backward()\n#   scaler.step(optimizer); scaler.update()")
        print("  [+] FP16 mixed precision recommendation added")

# 4. DataLoader
has_num_workers = "num_workers" in original
if not has_num_workers:
    workers = min(cpu_count, 8)
    header += f"#   - DataLoader: num_workers={workers}, pin_memory=True\n"
    notes.append(f"# Add to your DataLoader:\n#   num_workers={workers}, pin_memory=True, persistent_workers=True, prefetch_factor=2")
    print(f"  [+] DataLoader: num_workers={workers}, pin_memory=True")

header += "#\n\n"

# Build output
wrapper = header
wrapper += "# ── Zernel optimizations ──\n"
for opt in optimizations:
    wrapper += opt + "\n"
wrapper += "\n"

for note in notes:
    wrapper += note + "\n\n"

wrapper += "# ── Original script ──\n"
wrapper += original

with open(output_path, 'w') as f:
    f.write(wrapper)

print(f"\nOptimized script written to: {output_path}")
print(f"Run: python3 {output_path}")
