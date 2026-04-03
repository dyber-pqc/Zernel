# Copyright (C) 2026 Dyber, Inc. — Proprietary
#
# zernel-run: Run any Python training script with automatic optimizations.
#
# Usage:
#   zernel-run train.py [args...]
#   zernel-run --no-amp train.py
#   zernel-run --no-compile --verbose train.py
#   ZERNEL_AMP=0 zernel-run train.py

import sys
import os
import runpy


def main():
    args = sys.argv[1:]

    # Parse zernel flags (before the script path)
    script_args = []
    script_path = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--no-amp":
            os.environ["ZERNEL_AMP"] = "0"
        elif arg == "--no-compile":
            os.environ["ZERNEL_COMPILE"] = "0"
        elif arg == "--no-tf32":
            os.environ["ZERNEL_TF32"] = "0"
        elif arg == "--verbose" or arg == "-v":
            os.environ["ZERNEL_VERBOSE"] = "1"
        elif arg == "--quiet":
            os.environ["ZERNEL_REPORT"] = "0"
        elif arg.startswith("--grad-accum="):
            os.environ["ZERNEL_GRAD_ACCUM"] = arg.split("=", 1)[1]
        elif arg == "--help" or arg == "-h":
            _print_help()
            return
        elif not arg.startswith("-"):
            script_path = arg
            script_args = args[i:]
            break
        else:
            # Unknown flag — pass through to the script
            script_path = arg
            script_args = args[i:]
            break
        i += 1

    if script_path is None:
        _print_help()
        return

    if not os.path.exists(script_path):
        print(f"Error: script not found: {script_path}")
        sys.exit(1)

    # Apply Zernel optimizations before running the script
    # This patches torch at import time
    import zernel_runtime  # noqa: F401

    # Run the user's script
    sys.argv = script_args
    runpy.run_path(script_path, run_name="__main__")


def _print_help():
    print("zernel-run — Run training scripts with automatic optimization")
    print()
    print("Usage: zernel-run [options] script.py [script args...]")
    print()
    print("Options:")
    print("  --no-amp        Disable automatic mixed precision")
    print("  --no-compile    Disable torch.compile() wrapping")
    print("  --no-tf32       Disable TF32 matmul/cuDNN")
    print("  --verbose, -v   Show applied optimizations")
    print("  --quiet         Suppress end-of-training report")
    print("  --grad-accum=N  Enable gradient accumulation (N micro-steps)")
    print()
    print("Environment variables:")
    print("  ZERNEL_AMP=0/1          Toggle auto mixed precision")
    print("  ZERNEL_COMPILE=0/1      Toggle torch.compile()")
    print("  ZERNEL_TF32=0/1         Toggle TF32")
    print("  ZERNEL_AMP_DTYPE=bf16   Set AMP dtype (bf16 or fp16)")
    print("  ZERNEL_VERBOSE=1        Verbose output")
    print()
    print("Example:")
    print("  zernel-run train.py --epochs 10 --lr 3e-4")
    print("  zernel-run --verbose --no-compile train.py")


if __name__ == "__main__":
    main()
