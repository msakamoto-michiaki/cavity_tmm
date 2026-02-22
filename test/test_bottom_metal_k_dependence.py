# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from pathlib import Path
from utils import ensure_import_paths, write_json, is_verbose, summarize_real

def run(verbose: bool | None = None) -> list[str]:
    ensure_import_paths()
    from oled_cavity_phase3p1_policyB import build_current_base, INTERNAL_ORDER
    import run_bottom_metal_PEC_proxy as bm

    n0, d0 = build_current_base()

    # minimal best geometry json (use base thicknesses)
    geom = {
        "lambda_nm": 620.0,
        "d_best_nm": {k: float(d0[k]) for k in INTERNAL_ORDER},
    }
    out = Path(__file__).resolve().parent / "_out" / "geom"
    out.mkdir(parents=True, exist_ok=True)
    best_json = out / "best_geometry.json"
    write_json(best_json, geom)

    lam_m = float(geom["lambda_nm"]) * 1e-9
    n_layers = {k: complex(n0[k]) for k in INTERNAL_ORDER}
    n_layers["EML"] = complex(np.real(n_layers["EML"]), 0.0)
    n_top = complex(n0["cathode1"])
    d_layers_nm = geom["d_best_nm"]
    eml_index = 1 + INTERNAL_ORDER.index("EML")
    z_ex_m = 0.5 * d_layers_nm["EML"] * 1e-9

    k_list = [50.0, 200.0, 800.0, 2000.0]
    ratios = []
    if is_verbose(verbose):
        print("[TEST] bottom-metal k dependence (boundary |E|^2 suppression)")
        print("  evaluates ratio(k)=|E(z=0)|/max|E(EML)| using run_bottom_metal_PEC_proxy.compute_E0_over_emlmax")
        print(f"  lambda={geom['lambda_nm']} nm, u=0, TE-h")
        print(f"  k_list={k_list}")
        print("  criteria:")
        print("    (a) all ratios finite")
        print("    (b) ratio(k_max) < ratio(k_min)")
        print("    (c) loose decreasing trend: ratio(800) <= 1.2*ratio(200)")
        print()
    for k in k_list:
        ratio, rA, rB = bm.compute_E0_over_emlmax(
            n_metal=complex(0.14, k),
            n_top=n_top,
            n_layers=n_layers,
            d_layers_nm=d_layers_nm,
            eml_index=eml_index,
            z_ex_m=z_ex_m,
            lambda0_m=lam_m,
            u=0.0,
            pol="TE",
            orientation="h",
        )
        ratios.append(float(ratio))

        if is_verbose(verbose):
            print(f"  k={k:7.1f}  ratio={float(ratio):.6g}  rA={rA:.6g}  rB={rB:.6g}")
    fails: list[str] = []
    if not np.all(np.isfinite(ratios)):
        fails.append(f"ratio(k) has non-finite: {ratios}")
        return fails
    if not (ratios[-1] < ratios[0]):
        fails.append(f"ratio(k) did not improve from k={k_list[0]} to k={k_list[-1]}: {ratios}")
    # loose monotonic trend check
    if not (ratios[2] <= ratios[1] * 1.2):
        fails.append(f"ratio(k) not decreasing trend (k=200->800): {ratios}")

    if is_verbose(verbose):
        print(f"  ratios: {summarize_real(ratios)}")
        print()
    return fails
