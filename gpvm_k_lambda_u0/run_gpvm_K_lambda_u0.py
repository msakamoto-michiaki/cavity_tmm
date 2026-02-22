#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict-EML GPVM: K(λ,u=0) with full polarization decomposition.

Computes (strict EML interface r_A/r_B):
  - K_TE_h(λ,u=0)  (Eq.26+Eq.30)
  - K_TM_h(λ,u=0)  (Eq.27+Eq.30)
  - K_TM_v(λ,u=0)  (Eq.28+Eq.30)  (should be ~0 at u=0)
and outputs:
  - K_iso      = (2/3)K_TE_h + (2/3)K_TM_h + (1/3)K_TM_v
  - K_TE_only  = (2/3)K_TE_h
  - K_TM_h_only= (2/3)K_TM_h
  - K_TM_v_only= (1/3)K_TM_v

Uses the phase3 best geometry at:
  out_gpvm_step2/best_geometry.json

Also replaces ITO index by PEC-like metal:
  n_ITO = 0.14 + 2000 i

Run:
  python gpvm_k_lambda_u0/run_gpvm_K_lambda_u0.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import argparse
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent  # repo root
sys.path.insert(0, str(BASE))

from common.oled_cavity_phase3p1_policyB import build_current_base
from common.units import nm_to_m
from gpvm.kz import kz
from gpvm.system_matrix import stack_transfer_matrix, rt_from_transfer_matrix

BEST_JSON = BASE / "out_gpvm_step2" / "best_geometry.json"


from gpvm.eqs_gpvm import Pe_TE_h_eq26, Pe_TM_h_eq27, Pe_TM_v_eq28, K_from_Pe_eq30

def build_n0_with_ito_as_metal() -> Dict:
    n0, _ = build_current_base()
    n0 = dict(n0)
    n0["ITO"] = 0.14 + 1j * 2000.0
    return n0


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict-EML GPVM: K(lambda,u=0)")
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
    d_best_nm = {k: float(v) for k, v in best["d_best_nm"].items()}

    n0 = build_n0_with_ito_as_metal()
    n_e = complex(float(np.real(n0["EML"])), 0.0)

    d_by_name_m = {k: nm_to_m(v) for k, v in d_best_nm.items()}
    d_eml_m = nm_to_m(d_best_nm["EML"])
    z_ex_m = nm_to_m(0.5 * d_best_nm["EML"])

    n_left = [n0["anode"], n0["ITO"], n0["pHTL"], n0["Rprime"], n0["HTL"], n0["EBL"], n_e]
    d_left = [
        d_by_name_m["ITO"],
        d_by_name_m["pHTL"],
        d_by_name_m["Rprime"],
        d_by_name_m["HTL"],
        d_by_name_m["EBL"],
    ]
    n_right = [n_e, n0["ETL"], n0["cathode1"], n0["cathode"], n0["CPL"], n0["air"]]
    d_right = [
        d_by_name_m["ETL"],
        d_by_name_m["cathode1"],
        d_by_name_m["cathode"],
        d_by_name_m["CPL"],
    ]
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

    K_TE_h = np.zeros_like(lam_nm)
    K_TM_h = np.zeros_like(lam_nm)
    K_TM_v = np.zeros_like(lam_nm)

    for i, lm in enumerate(lam_m):
        kz_e = kz(n_e, n_e, u, float(lm))

        # TE
        M_left_te = stack_transfer_matrix(n_left, d_left, n_e=n_e, u=u, lambda0_m=float(lm), pol="TE")
        rA_te = -M_left_te[0, 1] / M_left_te[0, 0]
        M_right_te = stack_transfer_matrix(n_right, d_right, n_e=n_e, u=u, lambda0_m=float(lm), pol="TE")
        rB_te, _ = rt_from_transfer_matrix(M_right_te)
        Pe_te = Pe_TE_h_eq26(rA_te, rB_te, kz_e, d_eml_m, z_ex_m)
        K_TE_h[i] = K_from_Pe_eq30(Pe_te, u)

        # TM
        M_left_tm = stack_transfer_matrix(n_left, d_left, n_e=n_e, u=u, lambda0_m=float(lm), pol="TM")
        rA_tm = -M_left_tm[0, 1] / M_left_tm[0, 0]
        M_right_tm = stack_transfer_matrix(n_right, d_right, n_e=n_e, u=u, lambda0_m=float(lm), pol="TM")
        rB_tm, _ = rt_from_transfer_matrix(M_right_tm)
        Pe_tmh = Pe_TM_h_eq27(rA_tm, rB_tm, kz_e, d_eml_m, z_ex_m, u)
        Pe_tmv = Pe_TM_v_eq28(rA_tm, rB_tm, kz_e, d_eml_m, z_ex_m, u)
        K_TM_h[i] = K_from_Pe_eq30(Pe_tmh, u)
        K_TM_v[i] = K_from_Pe_eq30(Pe_tmv, u)

    K_TE_only = (2.0 / 3.0) * K_TE_h
    K_TM_h_only = (2.0 / 3.0) * K_TM_h
    K_TM_v_only = (1.0 / 3.0) * K_TM_v
    K_iso = K_TE_only + K_TM_h_only + K_TM_v_only

    out_dir_in = Path(args.outdir).expanduser()
    out_dir = out_dir_in if out_dir_in.is_absolute() else (BASE / out_dir_in)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"lam_nm{args.tag}.npy", lam_nm)
    np.save(out_dir / f"K_TE_h{args.tag}.npy", K_TE_h)
    np.save(out_dir / f"K_TM_h{args.tag}.npy", K_TM_h)
    np.save(out_dir / f"K_TM_v{args.tag}.npy", K_TM_v)
    np.save(out_dir / f"K_TE_only{args.tag}.npy", K_TE_only)
    np.save(out_dir / f"K_TM_h_only{args.tag}.npy", K_TM_h_only)
    np.save(out_dir / f"K_TM_v_only{args.tag}.npy", K_TM_v_only)
    np.save(out_dir / f"K_iso{args.tag}.npy", K_iso)

    # Plot (normalized)
    def norm(x):
        return x / (np.max(x) + 1e-30)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(lam_nm, norm(K_iso), label="K_iso")
    ax.plot(lam_nm, norm(K_TE_only), label="K_TE_only (2/3 TE_h)")
    ax.plot(lam_nm, norm(K_TM_h_only), label="K_TM_h_only (2/3 TM_h)")
    ax.plot(lam_nm, norm(K_TM_v_only), label="K_TM_v_only (1/3 TM_v)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized K")
    ax.set_title("Strict-EML GPVM: K(λ,u=0) components (normalized)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"gpvm_K_lambda_u0_components_strict{args.tag}.png", dpi=220)
    plt.close(fig)

    print("[gpvm_k_lambda_u0] wrote:")
    print("  -", out_dir / f"gpvm_K_lambda_u0_components_strict{args.tag}.png")


if __name__ == "__main__":
    main()
