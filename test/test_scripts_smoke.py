# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import numpy as np
from utils import ensure_import_paths, run_subprocess, temp_dir, is_verbose, summarize_real

def run(verbose: bool | None = None) -> list[str]:
    ensure_import_paths()
    root = Path(__file__).resolve().parents[1]

    out = temp_dir("smoke_")
    # Writer (optimization) - run once
    geom_dir = out / "out_gpvm_step2" / "G"
    geom_json = geom_dir / "best_geometry.json"
    cmd = ["python", str(root/"run_phase3_opt_then_gpvm_eml_profile.py"),
           "--lambda-nm", "530", "--best-json", str(geom_json)]
    if is_verbose(verbose):
        print("[TEST] scripts smoke (subset of run_rgb_full_eval.sh)")
        print(f"  temp outdir: {out}")
        print("  steps: WRITER -> U0_MAIN -> U0_AVSB -> BOTTOM_METAL -> HEATMAP(u0-only)")
        print()
        print("[SMOKE] WRITER command:")
        print("  ", " ".join(cmd))
    code, stdout, stderr = run_subprocess(cmd, cwd=root, timeout_s=300)
    fails: list[str] = []
    if code != 0:
        fails.append(f"WRITER failed (exit={code}). stderr tail:\n{stderr[-800:]}")
        return fails
    if is_verbose(verbose):
        print("  WRITER exit=0")
        if stderr.strip():
            print("  WRITER stderr (tail):")
            print(stderr[-400:])
        print()
    # expected proxy outputs (fixed names) in geom_dir
    expect_files = [
        geom_json,
        geom_dir/"E2_proxy_lcav_phase3BT_raw.npy",
        geom_dir/"gpvm_lcav_proxy_phase3BT.png",
    ]
    for p in expect_files:
        if not p.exists():
            fails.append(f"WRITER missing output: {p}")
    if is_verbose(verbose):
        print("  WRITER expected outputs:")
        for p in expect_files:
            print(f"   - {'OK' if p.exists() else 'MISSING'}  {p}")
        print()

    # u0 main and avsb with small lambda grid
    out_u0 = out / "u0"
    out_u0.mkdir(parents=True, exist_ok=True)
    cmd = ["python", str(root/"gpvm_k_lambda_u0/run_gpvm_K_lambda_u0.py"),
           "--best-json", str(geom_json), "--outdir", str(out_u0), "--tag", "_G",
           "--lam-span-nm", "60", "--n-lam", "31"]
    if is_verbose(verbose):
        print("[SMOKE] U0_MAIN command:")
        print("  ", " ".join(cmd))
    code, _, err = run_subprocess(cmd, cwd=root, timeout_s=180)
    if code != 0:
        fails.append(f"U0_MAIN failed (exit={code}). stderr tail:\n{err[-800:]}")
    else:
        if not (out_u0/"gpvm_K_lambda_u0_components_strict_G.png").exists():
            fails.append("U0_MAIN missing output png")
        if is_verbose(verbose):
            print("  U0_MAIN exit=0")
            print(f"  expected png: {out_u0/'gpvm_K_lambda_u0_components_strict_G.png'}")
            print()

    cmd = ["python", str(root/"gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py"),
           "--best-json", str(geom_json), "--outdir", str(out_u0), "--tag", "_G",
           "--lam-span-nm", "60", "--n-lam", "31"]
    if is_verbose(verbose):
        print("[SMOKE] U0_AVSB command:")
        print("  ", " ".join(cmd))
    code, _, err = run_subprocess(cmd, cwd=root, timeout_s=180)
    if code != 0:
        fails.append(f"U0_AVSB failed (exit={code}). stderr tail:\n{err[-800:]}")
    else:
        if not (out_u0/"gpvm_K_lambda_u0_A_vs_B_G.png").exists():
            fails.append("U0_AVSB missing output png")
        if is_verbose(verbose):
            print("  U0_AVSB exit=0")
            print(f"  expected png: {out_u0/'gpvm_K_lambda_u0_A_vs_B_G.png'}")
            print()

    # bottom metal script
    out_bm = out / "bm"
    out_bm.mkdir(parents=True, exist_ok=True)
    cmd = ["python", str(root/"run_bottom_metal_PEC_proxy.py"),
           "--best-json", str(geom_json), "--outdir", str(out_bm), "--tag", "_G"]
    if is_verbose(verbose):
        print("[SMOKE] BOTTOM_METAL command:")
        print("  ", " ".join(cmd))
    code, _, err = run_subprocess(cmd, cwd=root, timeout_s=240)
    if code != 0:
        fails.append(f"BOTTOM_METAL failed (exit={code}). stderr tail:\n{err[-800:]}")
    else:
        if not (out_bm/"need_k_for_E0_small_G.png").exists():
            fails.append("BOTTOM_METAL missing need_k png")
        if not (out_bm/"gpvm_lcav_profile_bottomMetal_k2000_G.png").exists():
            fails.append("BOTTOM_METAL missing profile png")
        if is_verbose(verbose):
            print("  BOTTOM_METAL exit=0")
            print(f"  expected: {out_bm/'need_k_for_E0_small_G.png'}")
            print(f"  expected: {out_bm/'gpvm_lcav_profile_bottomMetal_k2000_G.png'}")
            print()

    # heatmap: validate u0-only computation using the same best-json
    try:
        import gpvm_K_lambda_kpar_map_strict.run_gpvm_K_lambda_kpar_map_strict as hm
        from pathlib import Path as _Path
        hm.BEST_JSON_PATH = _Path(str(geom_json))
        lam_nm = np.linspace(500, 560, 11)
        d = hm.compute_u0_only(lam_nm)
        if is_verbose(verbose):
            print("[SMOKE] HEATMAP u0-only (no 2D map):")
            print(f"  lam_nm grid: {lam_nm.tolist()}")
        for key in ["K_iso_u0", "K_TE_only_u0", "K_TM_h_only_u0", "K_TM_v_only_u0"]:
            if key not in d:
                fails.append(f"HEATMAP compute_u0_only missing key {key}")
            else:
                arr = np.asarray(d[key])
                if arr.shape != lam_nm.shape:
                    fails.append(f"HEATMAP {key} shape mismatch: {arr.shape} vs {lam_nm.shape}")
                if is_verbose(verbose):
                    print(f"  {key}: shape={arr.shape}, min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}")
        if is_verbose(verbose):
            print()
    except Exception as e:
        fails.append(f"HEATMAP u0-only smoke failed: {e}")

    return fails
