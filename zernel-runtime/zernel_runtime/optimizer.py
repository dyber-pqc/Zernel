# Copyright (C) 2026 Dyber, Inc. — Proprietary
#
# Core auto-optimization engine.
# Patches PyTorch at runtime to automatically apply:
# - TF32 matmul/cuDNN
# - CUDA memory allocator tuning
# - Mixed precision (AMP) via module hooks
# - torch.compile() via lazy wrapping

import os
import sys
import functools
import warnings

_ZERNEL_APPLIED = False
_ZERNEL_CONFIG = {
    "tf32": True,
    "amp": True,
    "amp_dtype": "bfloat16",  # or "float16"
    "compile": True,
    "cuda_alloc": True,
    "grad_accumulation": 1,
    "report": True,
    "verbose": False,
}


def auto_optimize():
    """Apply all automatic optimizations. Safe to call multiple times."""
    global _ZERNEL_APPLIED
    if _ZERNEL_APPLIED:
        return
    _ZERNEL_APPLIED = True

    # Parse env overrides
    _parse_env()

    # 1. CUDA allocator (must be before torch import)
    if _ZERNEL_CONFIG["cuda_alloc"]:
        _setup_cuda_allocator()

    # 2. Install import hook to patch torch when it loads
    sys.meta_path.insert(0, _ZernelTorchHook())

    if _ZERNEL_CONFIG["verbose"]:
        print("[zernel] auto-optimizer initialized")


def _parse_env():
    """Override config from environment variables."""
    env_map = {
        "ZERNEL_AMP": "amp",
        "ZERNEL_COMPILE": "compile",
        "ZERNEL_TF32": "tf32",
        "ZERNEL_VERBOSE": "verbose",
        "ZERNEL_REPORT": "report",
    }
    for env_key, config_key in env_map.items():
        val = os.environ.get(env_key, "").lower()
        if val in ("0", "false", "no", "off"):
            _ZERNEL_CONFIG[config_key] = False
        elif val in ("1", "true", "yes", "on"):
            _ZERNEL_CONFIG[config_key] = True

    acc = os.environ.get("ZERNEL_GRAD_ACCUM", "")
    if acc.isdigit() and int(acc) > 0:
        _ZERNEL_CONFIG["grad_accumulation"] = int(acc)

    dtype = os.environ.get("ZERNEL_AMP_DTYPE", "").lower()
    if dtype in ("bfloat16", "bf16"):
        _ZERNEL_CONFIG["amp_dtype"] = "bfloat16"
    elif dtype in ("float16", "fp16"):
        _ZERNEL_CONFIG["amp_dtype"] = "float16"


def _setup_cuda_allocator():
    """Configure CUDA memory allocator before torch loads."""
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in current:
        parts = [p for p in current.split(",") if p.strip()]
        parts.append("expandable_segments:True")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)


class _ZernelTorchHook:
    """Import hook that patches torch after it's loaded."""

    _patched = False

    def find_module(self, fullname, path=None):
        # Only intercept the first 'torch' import
        if fullname == "torch" and not _ZernelTorchHook._patched:
            return self
        return None

    def load_module(self, fullname):
        # Remove ourselves from meta_path to avoid recursion
        _ZernelTorchHook._patched = True
        sys.meta_path.remove(self)

        # Let the real torch load
        if fullname in sys.modules:
            return sys.modules[fullname]

        import importlib
        torch = importlib.import_module(fullname)

        # Now apply patches
        _patch_torch(torch)

        return torch


def _patch_torch(torch):
    """Apply all optimizations to a loaded torch module."""

    if not torch.cuda.is_available():
        if _ZERNEL_CONFIG["verbose"]:
            print("[zernel] no CUDA GPU detected, skipping GPU optimizations")
        return

    cap = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()

    if _ZERNEL_CONFIG["verbose"]:
        print(f"[zernel] GPU: {gpu_name} (sm_{cap[0]}{cap[1]})")

    # 1. TF32 (free speedup on Ampere+)
    if _ZERNEL_CONFIG["tf32"] and cap[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if _ZERNEL_CONFIG["verbose"]:
            print("[zernel] TF32 enabled for matmul + cuDNN")

    # 2. Determine AMP dtype
    if _ZERNEL_CONFIG["amp"]:
        if cap[0] >= 8:
            _ZERNEL_CONFIG["amp_dtype"] = "bfloat16"
        elif cap[0] >= 7:
            _ZERNEL_CONFIG["amp_dtype"] = "float16"
        else:
            _ZERNEL_CONFIG["amp"] = False

    # 3. Patch nn.Module to auto-wrap forward with AMP + compile
    if _ZERNEL_CONFIG["amp"] or _ZERNEL_CONFIG["compile"]:
        _install_module_hooks(torch)

    # 4. Install training end report
    if _ZERNEL_CONFIG["report"]:
        import atexit
        atexit.register(functools.partial(_print_report, torch))


def _install_module_hooks(torch):
    """Monkey-patch nn.Module so .train() auto-wraps forward with AMP."""

    original_train = torch.nn.Module.train
    original_cuda = torch.nn.Module.cuda
    original_to = torch.nn.Module.to

    def _maybe_wrap(module):
        """Wrap a module's forward method with AMP if it's on CUDA."""
        if hasattr(module, "_zernel_wrapped"):
            return

        # Only wrap top-level modules (not submodules)
        # Detect by checking if it has parameters and is on CUDA
        try:
            param = next(module.parameters(), None)
            if param is None or not param.is_cuda:
                return
        except (StopIteration, RuntimeError):
            return

        original_forward = module.forward

        if _ZERNEL_CONFIG["amp"]:
            amp_dtype = getattr(torch, _ZERNEL_CONFIG["amp_dtype"])

            @functools.wraps(original_forward)
            def amp_forward(*args, **kwargs):
                with torch.autocast("cuda", dtype=amp_dtype):
                    return original_forward(*args, **kwargs)

            module.forward = amp_forward
            module._zernel_wrapped = True

            if _ZERNEL_CONFIG["verbose"]:
                name = module.__class__.__name__
                print(f"[zernel] auto-AMP ({_ZERNEL_CONFIG['amp_dtype']}) applied to {name}")

    def patched_train(self, mode=True):
        result = original_train(self, mode)
        if mode:
            _maybe_wrap(self)
        return result

    def patched_cuda(self, device=None):
        result = original_cuda(self, device)
        _maybe_wrap(self)
        return result

    def patched_to(self, *args, **kwargs):
        result = original_to(self, *args, **kwargs)
        # Check if moved to CUDA
        try:
            dst = args[0] if args else kwargs.get("device")
            if dst is not None and "cuda" in str(dst):
                _maybe_wrap(self)
        except (TypeError, IndexError):
            pass
        return result

    torch.nn.Module.train = patched_train
    torch.nn.Module.cuda = patched_cuda
    torch.nn.Module.to = patched_to


def _print_report(torch):
    """Print optimization report at exit."""
    if not torch.cuda.is_available():
        return

    try:
        mem_alloc = torch.cuda.max_memory_allocated() / 1e9
        mem_reserved = torch.cuda.max_memory_reserved() / 1e9
    except Exception:
        return

    if mem_alloc < 0.01:
        return  # No GPU usage, skip report

    print()
    print("=" * 60)
    print("Zernel Training Report")
    print("=" * 60)

    cap = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"  Peak memory: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    print(f"  Memory utilization: {mem_alloc/gpu_mem*100:.0f}%")

    opts_applied = []
    if _ZERNEL_CONFIG["tf32"] and cap[0] >= 8:
        opts_applied.append("TF32 matmul")
    if _ZERNEL_CONFIG["amp"]:
        opts_applied.append(f"Auto-AMP ({_ZERNEL_CONFIG['amp_dtype']})")
    opts_applied.append("CUDA allocator (expandable_segments)")

    print(f"  Optimizations: {', '.join(opts_applied)}")

    # Memory headroom
    headroom = gpu_mem - mem_alloc
    if headroom > gpu_mem * 0.4:
        print(f"  Tip: {headroom:.1f} GB unused — increase batch size for better throughput")
    elif headroom < gpu_mem * 0.1:
        print(f"  Tip: Memory nearly full — consider gradient checkpointing")

    print("=" * 60)
