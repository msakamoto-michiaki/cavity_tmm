#!/usr/bin/env python3
"""Bottom metal PEC-proxy diagnostics.

Creates two figures used in the discussion:

(A) How large Im(n)=k is needed so that |E(z=0)|/max|E(EML)| < 1e-3.
(B) LCAV |E|^2 profile for a representative PEC-like choice (default k=2000).

Model:
- TE polarization
- u=0 (normal incidence)
- dipole orientation = h (horizontal) for TE×h channel
- EML is treated lossless (n = Re(n)+0j), consistent with phase3/GPVM convention.

Bottom boundary:
- Semi-infinite metal ambient with n = n_re + i*k.
  (This is the numerically stable way to create a near-PEC boundary without enforcing it.)

Outputs:
  ./out_bottom_metal/
    need_k_for_E0_small.png
    need_k_for_E0_small.csv
    gpvm_lcav_profile_bottomMetal_k2000.png

"""

from __future__ import annotations

import json
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from common.oled_cavity_phase3p1_policyB import build_current_base, INTERNAL_ORDER

from gpvm.kz import kz as kz_layer
from gpvm.matrices import interface_matrix, layer_matrix
from gpvm.source_terms import source_terms
from gpvm.source_plane import solve_source_plane_fields
from gpvm.system_matrix import build_system_matrices_SA_SB, eq12_rt_from_SA_SB


@dataclass
class Config:
    n_re: float = 0.14
    d_metal_nm: float = 200.0  # kept for documentation; semi-infinite ambient ignores it
    k_list: Tuple[float, ...] = (30, 50, 80, 120, 200, 300, 500, 800, 1200, 2000, 4000, 8000)
    k_profile: float = 2000.0
    thr_amp: float = 1e-3
    u: float = 0.0
    pol: str = "TE"
    orientation: str = "h"


def _load_best_geometry(best_json: str) -> Dict:
    with open(best_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _max_abs_in_eml(fields, kz_eml: complex, d_eml_m: float, z_ex_m: float) -> float:
    # Piecewise field reconstruction within EML around the source plane.
    x = np.linspace(0.0, d_eml_m, 2001)
    E = np.zeros_like(x, dtype=complex)

    maskL = x <= z_ex_m
    xl = x[maskL] - z_ex_m
    E[maskL] = fields.Ea_plus * np.exp(1j * kz_eml * xl) + fields.Ea_minus * np.exp(-1j * kz_eml * xl)

    maskR = ~maskL
    xr = x[maskR] - z_ex_m
    E[maskR] = fields.Eb_plus * np.exp(1j * kz_eml * xr) + fields.Eb_minus * np.exp(-1j * kz_eml * xr)

    return float(np.max(np.abs(E)) + 1e-30)


def compute_E0_over_emlmax(
    *,
    n_metal: complex,
    n_top: complex,
    n_layers: Dict[str, complex],
    d_layers_nm: Dict[str, float],
    eml_index: int,
    z_ex_m: float,
    lambda0_m: float,
    u: float,
    pol: str,
    orientation: str,
) -> Tuple[float, complex, complex]:
    """Return (|E0|/max|E_eml|, rA, rB) for TE×h at u=0.

    E0 is evaluated at z=0 at the pHTL left boundary (inside the cavity).
    """

    # Lossless EML
    n_e = complex(np.real(n_layers["EML"]), 0.0)

    n_list = [n_metal] + [n_layers[k] if k != "EML" else n_e for k in INTERNAL_ORDER] + [n_top]
    d_list_m = [d_layers_nm[k] * 1e-9 for k in INTERNAL_ORDER]

    d_eml_m = d_layers_nm["EML"] * 1e-9

    SA, SB = build_system_matrices_SA_SB(
        n_list=n_list,
        d_list_m=d_list_m,
        eml_index=eml_index,
        z_ex_m=z_ex_m,
        n_e=n_e,
        u=u,
        lambda0_m=lambda0_m,
        pol=pol,
    )
    rA, tA, rB, tB = eq12_rt_from_SA_SB(SA, SB)

    kz_eml = kz_layer(n_e, n_e, u, lambda0_m)
    st = source_terms(u=u, pol=pol, orientation=orientation)
    fields = solve_source_plane_fields(
        A_plus=st.A_plus,
        A_minus=st.A_minus,
        rA=rA,
        rB=rB,
        kz_e=kz_eml,
        d_e_m=d_eml_m,
        z_ex_m=z_ex_m,
    )

    # Field at z=0 (pHTL-left boundary, inside pHTL)
    # We obtain amplitudes at the metal/pHTL interface from SA, then convert to pHTL side.
    v_metal = SA @ np.array([fields.Ea_plus, fields.Ea_minus], dtype=complex)
    I_mp = interface_matrix(n_metal, complex(n_layers["pHTL"]), n_e, u, lambda0_m, pol)
    v_phtl = np.linalg.solve(I_mp, v_metal)
    Ep, Em = v_phtl
    E0 = Ep + Em

    max_eml = _max_abs_in_eml(fields, kz_eml, d_eml_m, z_ex_m)
    return float(abs(E0) / max_eml), rA, rB


def build_lcav_profile(
    *,
    n_metal: complex,
    n_top: complex,
    n_layers: Dict[str, complex],
    d_layers_nm: Dict[str, float],
    eml_index: int,
    z_ex_m: float,
    lambda0_m: float,
    u: float,
    pol: str,
    orientation: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (z_nm, E2_norm, E0_amp_norm) across Lcav for TE×h."""

    n_e = complex(np.real(n_layers["EML"]), 0.0)

    n_list = [n_metal] + [n_layers[k] if k != "EML" else n_e for k in INTERNAL_ORDER] + [n_top]
    d_list_m = [d_layers_nm[k] * 1e-9 for k in INTERNAL_ORDER]

    d_eml_m = d_layers_nm["EML"] * 1e-9

    SA, SB = build_system_matrices_SA_SB(
        n_list=n_list,
        d_list_m=d_list_m,
        eml_index=eml_index,
        z_ex_m=z_ex_m,
        n_e=n_e,
        u=u,
        lambda0_m=lambda0_m,
        pol=pol,
    )
    rA, tA, rB, tB = eq12_rt_from_SA_SB(SA, SB)

    kz_eml = kz_layer(n_e, n_e, u, lambda0_m)
    st = source_terms(u=u, pol=pol, orientation=orientation)
    fields = solve_source_plane_fields(
        A_plus=st.A_plus,
        A_minus=st.A_minus,
        rA=rA,
        rB=rB,
        kz_e=kz_eml,
        d_e_m=d_eml_m,
        z_ex_m=z_ex_m,
    )

    # z grid within Lcav
    layer_order = INTERNAL_ORDER[:]  # pHTL..ETL
    th_nm = [d_layers_nm[name] for name in layer_order]
    bounds = np.cumsum([0.0] + th_nm)
    z_nm = np.linspace(0.0, bounds[-1], 7001)

    # Left-side: source plane -> EML left boundary
    v_left = {}
    v_eml_left = layer_matrix(n_e, n_e, u, lambda0_m, z_ex_m) @ np.array([fields.Ea_plus, fields.Ea_minus], dtype=complex)
    v_left["EML"] = v_eml_left

    prev = "EML"
    v_prev_left = v_eml_left
    for name in ["EBL", "HTL", "Rprime", "pHTL"]:
        I = interface_matrix(complex(n_layers[name]), complex(n_layers[prev] if prev != "EML" else n_e), n_e, u, lambda0_m, pol)
        v_right = I @ v_prev_left
        L = layer_matrix(complex(n_layers[name]), n_e, u, lambda0_m, d_layers_nm[name] * 1e-9)
        v_left[name] = L @ v_right
        prev = name
        v_prev_left = v_left[name]

    # Right-side: source plane -> EML right boundary -> ETL
    v_right_left = {}
    Lseg = layer_matrix(n_e, n_e, u, lambda0_m, (d_eml_m - z_ex_m))
    v_eml_right = np.linalg.solve(Lseg, np.array([fields.Eb_plus, fields.Eb_minus], dtype=complex))
    v_right_left["EML"] = np.array([fields.Eb_plus, fields.Eb_minus], dtype=complex)  # at source plane, right side

    I_eml_etl = interface_matrix(n_e, complex(n_layers["ETL"]), n_e, u, lambda0_m, pol)
    v_right_left["ETL"] = np.linalg.solve(I_eml_etl, v_eml_right)

    # Build field
    E = np.zeros_like(z_nm, dtype=complex)
    for i, name in enumerate(layer_order):
        z0, z1 = bounds[i], bounds[i + 1]
        mask = (z_nm >= z0) & (z_nm <= z1 + 1e-12)
        x_m = (z_nm[mask] - z0) * 1e-9
        kz_j = kz_layer(complex(n_layers[name]) if name != "EML" else n_e, n_e, u, lambda0_m)

        if name == "EML":
            z_ex_global = z0 + z_ex_m * 1e9
            maskL = mask & (z_nm <= z_ex_global)
            maskR = mask & (z_nm >= z_ex_global)

            xL = (z_nm[maskL] - z0) * 1e-9
            EpL, EmL = v_left["EML"]
            E[maskL] = EpL * np.exp(1j * kz_j * xL) + EmL * np.exp(-1j * kz_j * xL)

            xR = (z_nm[maskR] - z_ex_global) * 1e-9
            EpR, EmR = v_right_left["EML"]
            E[maskR] = EpR * np.exp(1j * kz_j * xR) + EmR * np.exp(-1j * kz_j * xR)

        elif name in ["pHTL", "Rprime", "HTL", "EBL"]:
            Ep, Em = v_left[name]
            E[mask] = Ep * np.exp(1j * kz_j * x_m) + Em * np.exp(-1j * kz_j * x_m)

        else:  # ETL
            Ep, Em = v_right_left["ETL"]
            E[mask] = Ep * np.exp(1j * kz_j * x_m) + Em * np.exp(-1j * kz_j * x_m)

    E2 = np.abs(E) ** 2
    E2_norm = E2 / (np.max(E2) + 1e-30)
    E0_amp_norm = float(np.sqrt(E2_norm[0]))
    return z_nm, E2_norm, E0_amp_norm


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="Bottom metal PEC-proxy diagnostics (uses best_geometry.json)")
    ap.add_argument("--best-json", type=str, default=os.path.join(root, "out_gpvm_step2", "best_geometry.json"),
                    help="Path to best geometry JSON (default: ./out_gpvm_step2/best_geometry.json)")
    ap.add_argument("--outdir", type=str, default=os.path.join(root, "out_bottom_metal"),
                    help="Output directory (default: ./out_bottom_metal)")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional suffix for output filenames, e.g. _R/_G/_B")
    ap.add_argument("--xmax-nm", type=float, default=None,
                    help="Override xmax for LCAV profile plot (nm). Useful to align RGB x-ranges.")
    args = ap.parse_args()

    best_json = args.best_json
    if not os.path.isabs(best_json):
        best_json = os.path.join(root, best_json)

    outdir = args.outdir
    if not os.path.isabs(outdir):
        outdir = os.path.join(root, outdir)
    os.makedirs(outdir, exist_ok=True)

    cfg = Config()

    geom = _load_best_geometry(best_json)
    lam_nm = float(geom.get("lambda_nm", 650.0))
    lam_m = lam_nm * 1e-9

    n0, _ = build_current_base()
    n_layers = {k: complex(n0[k]) for k in INTERNAL_ORDER}
    n_layers["EML"] = complex(np.real(n_layers["EML"]), 0.0)

    n_top = complex(n0["cathode1"])  # keep phase3 top

    d_layers_nm = {k: float(v) for k, v in geom["d_best_nm"].items()}

    # metal is semi-infinite ambient => eml_index shifts by +1
    eml_index = 1 + INTERNAL_ORDER.index("EML")
    d_eml_m = d_layers_nm["EML"] * 1e-9
    z_ex_m = 0.5 * d_eml_m

    # (A) scan k
    rows = []
    for k in cfg.k_list:
        n_metal = complex(cfg.n_re, float(k))
        E0_over_emlmax, rA, rB = compute_E0_over_emlmax(
            n_metal=n_metal,
            n_top=n_top,
            n_layers=n_layers,
            d_layers_nm=d_layers_nm,
            eml_index=eml_index,
            z_ex_m=z_ex_m,
            lambda0_m=lam_m,
            u=cfg.u,
            pol=cfg.pol,
            orientation=cfg.orientation,
        )
        rows.append((float(k), E0_over_emlmax, abs(1 + rA), abs(1 + rB)))

    rows = np.array(rows, dtype=float)

    csv_path = os.path.join(outdir, f"need_k_for_E0_small{args.tag}.csv")
    header = "k,|E0|/max|E_eml|,|1+rA|,|1+rB|"
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")

    fig_path = os.path.join(outdir, f"need_k_for_E0_small{args.tag}.png")
    plt.figure(figsize=(10.5, 5.6))
    plt.plot(rows[:, 0], rows[:, 1], marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(cfg.thr_amp, linestyle="--")
    plt.xlabel("bottom metal k in n=0.14 + i k (semi-infinite ambient)")
    plt.ylabel("|E(z=0 in pHTL)| / max(|E| over EML)  [GPVM TE×h]")
    plt.title(f"Required k for bottom node-like field @ λ={lam_nm:.0f} nm, u=0, TE×h")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)

    # first k meeting threshold
    meet = rows[rows[:, 1] < cfg.thr_amp]
    k_needed = float(meet[0, 0]) if len(meet) else None

    # (B) Lcav profile at k_profile
    n_metal = complex(cfg.n_re, float(cfg.k_profile))
    z_nm, E2_norm, E0_amp_norm = build_lcav_profile(
        n_metal=n_metal,
        n_top=n_top,
        n_layers=n_layers,
        d_layers_nm=d_layers_nm,
        eml_index=eml_index,
        z_ex_m=z_ex_m,
        lambda0_m=lam_m,
        u=cfg.u,
        pol=cfg.pol,
        orientation=cfg.orientation,
    )

    prof_path = os.path.join(outdir, f"gpvm_lcav_profile_bottomMetal_k2000{args.tag}.png")

    # mark EML and source plane
    layer_order = INTERNAL_ORDER[:]
    th_nm = [d_layers_nm[name] for name in layer_order]
    bounds = np.cumsum([0.0] + th_nm)
    idx_eml = layer_order.index("EML")
    z_eml0 = bounds[idx_eml]
    z_eml1 = bounds[idx_eml + 1]
    z_src = z_eml0 + z_ex_m * 1e9

    plt.figure(figsize=(10.5, 5.6))
    plt.plot(z_nm, E2_norm)
    if args.xmax_nm is not None:
        plt.xlim(0.0, float(args.xmax_nm))
    plt.axvline(z_eml0, linestyle="--")
    plt.axvline(z_eml1, linestyle="--")
    plt.axvline(z_src, linestyle="--")
    plt.xlabel("z in Lcav (nm)")
    plt.ylabel("normalized |E|^2 (max over Lcav = 1)")
    title = (
        f"LCAV |E|^2 profile (TE×h, u=0, λ={lam_nm:.0f} nm) with bottom metal k={cfg.k_profile:.0f}\n"
        f"|E(z=0)| ≈ {E0_amp_norm:.3e} (amplitude, normalized by max)"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(prof_path, dpi=180)

    # console summary
    print("[out_bottom_metal] wrote:")
    print(f"  - {os.path.relpath(fig_path, root)}")
    print(f"  - {os.path.relpath(csv_path, root)}")
    print(f"  - {os.path.relpath(prof_path, root)}")
    if k_needed is None:
        print(f"  threshold {cfg.thr_amp} NOT reached by k_list")
    else:
        print(f"  first k reaching |E0|/max|E_eml| < {cfg.thr_amp}: k={k_needed}")
    print("BEST_JSON path =", os.path.join(root, "out_gpvm_step2", "best_geometry.json"))
    print("INTERNAL_ORDER =", INTERNAL_ORDER)
    print("d_best_nm =", geom["d_best_nm"])
    Lcav = sum(float(geom["d_best_nm"][k]) for k in INTERNAL_ORDER)
    print("Lcav(sum INTERNAL_ORDER) =", Lcav)

if __name__ == "__main__":
    main()
