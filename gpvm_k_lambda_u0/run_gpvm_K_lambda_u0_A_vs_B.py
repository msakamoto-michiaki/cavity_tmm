#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate K(lambda,u=0) comparison:
(A) phase3-BT proxy vs (B) strict-EML GPVM.

- Uses GPVM Eq.(26) and Eq.(30) at u=0 (TE, horizontal dipole).
- Isotropic weighting: K_iso = (2/3) K_TE,h.
- Loads best geometry from --best-json (default: out_gpvm_step2/best_geometry.json).
- Replaces ITO index with: n = 0.14 + 2000i (PEC-like metal)

Run (from repo root):
  python gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py
  # or
  python gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py --best-json out_gpvm_step2/best_geometry_R.json --tag _R

Outputs are written into --outdir (default: gpvm_k_lambda_u0/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent  # repo root

sys.path.insert(0, str(BASE))

from common.oled_cavity_phase3p1_policyB import build_current_base
from common.tmm_rewrap_utils_policyB import terminal_reflections_BT
from common.units import nm_to_m
from gpvm.kz import kz
from gpvm.eqs_gpvm import Pe_TE_h_eq26, K_from_Pe_eq30
from gpvm.system_matrix import stack_transfer_matrix, rt_from_transfer_matrix


def K_TE_h_eq26(rA: complex, rB: complex, kz_e: complex, d_m: float, z_m: float, u_norm: float) -> float:
    """Eq.(26) -> Pe, then Eq.(30) -> K, specialized to TE×h.
    calculate Pe, and then K
    This is a thin wrapper around gpvm.eqs_gpvm.{Pe_TE_h_eq26, K_from_Pe_eq30}.
    """
    Pe = Pe_TE_h_eq26(rA=rA, rB=rB, kz_e=kz_e, d_eml_m=d_m, z_ex_m=z_m)
    return K_from_Pe_eq30(Pe=Pe, u=u_norm)

def norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return x / (np.max(x) + 1e-30)


def main() -> None:
    ap = argparse.ArgumentParser(description="K(lambda,u=0) comparison: phase3-BT proxy vs strict-EML GPVM (TE×h)")
    ap.add_argument("--best-json", type=str, default=str(BASE / "out_gpvm_step2" / "best_geometry.json"),
                    help="Path to best geometry JSON (default: ./out_gpvm_step2/best_geometry.json)")
    ap.add_argument("--outdir", type=str, default=str(HERE),
                    help="Output directory (default: this script directory)")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional suffix for output filenames, e.g. _R/_G/_B")
    ap.add_argument("--lam-center-nm", type=float, default=None,
                    help="Center wavelength (nm) for scan; default: best_geometry['lambda_nm']")
    ap.add_argument("--lam-span-nm", type=float, default=300.0,
                    help="Scan span (nm), used when min/max not fully specified (default: 300)")
    ap.add_argument("--lam-min-nm", type=float, default=None,
                    help="Override scan minimum wavelength (nm)")
    ap.add_argument("--lam-max-nm", type=float, default=None,
                    help="Override scan maximum wavelength (nm)")
    ap.add_argument("--n-lam", type=int, default=201,
                    help="Number of wavelength samples (default: 201)")
    args = ap.parse_args()

    best_json_in = Path(args.best_json).expanduser()
    best_json = best_json_in if best_json_in.is_absolute() else (BASE / best_json_in)
    best = json.loads(best_json.read_text(encoding="utf-8"))
    lam_target_nm = float(best.get("lambda_nm", 650.0))

    out_dir_in = Path(args.outdir).expanduser()
    out_dir = out_dir_in if out_dir_in.is_absolute() else (BASE / out_dir_in)
    out_dir.mkdir(parents=True, exist_ok=True)

    n0, _ = build_current_base()
    d_best = {k: float(v) for k, v in best["d_best_nm"].items()}

    n0_mod = dict(n0)
    n0_mod["ITO"] = 0.14 + 1j * 2000.0
    # wavelength grid
    lam0 = float(args.lam_center_nm) if args.lam_center_nm is not None else float(best.get('lambda_nm', 650.0))
    span = float(args.lam_span_nm)
    lam_min = lam0 - 0.5 * span
    lam_max = lam0 + 0.5 * span
    if args.lam_min_nm is not None:
        lam_min = float(args.lam_min_nm)
    if args.lam_max_nm is not None:
        lam_max = float(args.lam_max_nm)
    if not (lam_max > lam_min):
        lam_max = lam_min + 1.0
    n_lam = int(args.n_lam)
    if n_lam < 3:
        n_lam = 3
    lam_nm = np.linspace(lam_min, lam_max, n_lam)
    lam_m = lam_nm * 1e-9
    u = 0.0
    pol_TE = 0

    LBT_opt_m = nm_to_m(float(best["LBT_opt_nm"]))
    z_ex_opt_m = nm_to_m(float(best["z_ex_opt_nm"]))

    # (A) phase3 proxy: terminal reflections at B/T (ver15+: internal stable S-matrix)
    rb_arr, rt_arr, _ = terminal_reflections_BT(
        n0=n0_mod,
        d=d_best,
        lam_nm=lam_nm.astype(float),
        u=float(u),
        pol=int(pol_TE),
        phi_b_mode="pi",
        tm_top_pi_shift=True,
        tm_bottom_pi_shift=False,
    )
    rb_arr = np.asarray(rb_arr, complex)
    rt_arr = np.asarray(rt_arr, complex)

    K_A_te_h = np.zeros_like(lam_nm, dtype=float)
    for i, lm in enumerate(lam_m):
        k0 = 2.0 * np.pi / float(lm)
        # In proxy, kz treated as k0 in optical coordinate
        K_A_te_h[i] = K_TE_h_eq26(rb_arr[i], rt_arr[i], k0, LBT_opt_m, z_ex_opt_m, u)
    K_A_iso = (2.0 / 3.0) * K_A_te_h

    # (B) strict EML GPVM: rA/rB defined at EML interfaces
    d_by_name_m = {k: nm_to_m(v) for k, v in d_best.items()}
    n_e = complex(float(np.real(n0_mod["EML"])), 0.0)
    d_eml_m = nm_to_m(float(d_best["EML"]))
    z_ex_m = nm_to_m(0.5 * float(d_best["EML"]))

    def rA_rB_eml(lambda0_m: float, u_norm: float):
        n_left = [n0_mod["anode"], n0_mod["ITO"], n0_mod["pHTL"], n0_mod["Rprime"], n0_mod["HTL"], n0_mod["EBL"], n_e]
        d_left = [d_by_name_m["ITO"], d_by_name_m["pHTL"], d_by_name_m["Rprime"], d_by_name_m["HTL"], d_by_name_m["EBL"]]
        M_left = stack_transfer_matrix(n_left, d_left, n_e=n_e, u=u_norm, lambda0_m=lambda0_m, pol="TE")
        # incidence from the RIGHT onto the left stack: rA = -M12/M11 (Eq.12 for S^A)
        rA = -M_left[0, 1] / M_left[0, 0]

        n_right = [n_e, n0_mod["ETL"], n0_mod["cathode1"], n0_mod["cathode"], n0_mod["CPL"], n0_mod["air"]]
        d_right = [d_by_name_m["ETL"], d_by_name_m["cathode1"], d_by_name_m["cathode"], d_by_name_m["CPL"]]
        M_right = stack_transfer_matrix(n_right, d_right, n_e=n_e, u=u_norm, lambda0_m=lambda0_m, pol="TE")
        rB, _ = rt_from_transfer_matrix(M_right)
        return rA, rB

    K_B_te_h = np.zeros_like(lam_nm, dtype=float)
    for i, lm in enumerate(lam_m):
        rA, rB = rA_rB_eml(float(lm), u)
        kz_e = kz(n_e, n_e, u, float(lm))
        K_B_te_h[i] = K_TE_h_eq26(rA, rB, kz_e, d_eml_m, z_ex_m, u)
    K_B_iso = (2.0 / 3.0) * K_B_te_h

    fig_path = out_dir / f"gpvm_K_lambda_u0_A_vs_B{args.tag}.png"
    plt.figure(figsize=(7.0, 4.6))
    plt.plot(lam_nm, norm(K_A_iso), label="(A) phase3-BT proxy")
    plt.plot(lam_nm, norm(K_B_iso), label="(B) strict-EML GPVM")
    plt.axvline(lam_target_nm, linestyle="--", linewidth=1)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized K_iso")
    plt.title("K(λ, u=0) using Eq.(26)&(30) (TE,h → iso=2/3)\nBest geometry + ITO→metal(0.14+2000i)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    np.save(out_dir / f"lam_nm{args.tag}.npy", lam_nm)
    np.save(out_dir / f"K_A_iso{args.tag}.npy", K_A_iso)
    np.save(out_dir / f"K_B_iso{args.tag}.npy", K_B_iso)

    print("Wrote:", fig_path)


if __name__ == "__main__":
    main()
