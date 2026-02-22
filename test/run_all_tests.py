# -*- coding: utf-8 -*-
from __future__ import annotations
import importlib
from pathlib import Path
import inspect
import os
import sys

TESTS = [
    "test_fresnel_rewrap_vs_pymoosh",
    "test_gpvm_fresnel_sign_convention",
    "test_terminal_reflections_BT_vs_pymoosh",
    "test_gpvm_transfer_matrix_vs_pymoosh",
    "test_phase3_proxy_E2_consistency",
    "test_bottom_metal_k_dependence",
    "test_scripts_smoke",
]

def main() -> int:
    here = Path(__file__).resolve().parent
    fails_total = 0
    v = os.environ.get("TEST_VERBOSE", "1")
    print("=== ver17 test suite (verbose diagnostics) ===")
    print(f"TEST_VERBOSE={v}  (set to 0 to reduce prints)")
    for name in TESTS:
        print(f"\n--- {name} ---")
        mod = importlib.import_module(name)
        run_fn = getattr(mod, "run")
        # Call with verbose flag if supported
        try:
            if "verbose" in inspect.signature(run_fn).parameters:
                fails = run_fn(verbose=True)
            else:
                fails = run_fn()
        except TypeError:
            fails = run_fn()
        if fails:
            fails_total += len(fails)
            for f in fails:
                print("FAIL:", f)
        else:
            print("OK")
    print("\n=== summary ===")
    if fails_total:
        print(f"FAILED: {fails_total} issues")
        return 1
    print("ALL OK")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # When piping output (e.g. to `head`), stdout may close early.
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)
