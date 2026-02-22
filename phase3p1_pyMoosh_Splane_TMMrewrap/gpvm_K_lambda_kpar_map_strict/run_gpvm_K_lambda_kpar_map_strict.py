#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict-EML GPVM: K(λ,k//) heatmaps with full polarization decomposition.

This script computes, for strict-EML GPVM (EML interface r_A/r_B):
  - K_TE_h(λ,k//)   : TE polarization, horizontal dipole (Eq. 26 + Eq. 30)
  - K_TM_h(λ,k//)   : TM polarization, horizontal dipole (Eq. 27 + Eq. 30)
  - K_TM_v(λ,k//)   : TM polarization, vertical dipole   (Eq. 28 + Eq. 30)

and then outputs:
  - K_iso           = (2/3)K_TE_h + (2/3)K_TM_h + (1/3)K_TM_v
  - K_TE_only       = (2/3)K_TE_h
  - K_TM_h_only     = (2/3)K_TM_h
  - K_TM_v_only     = (1/3)K_TM_v

It also overlays region boundary lines (light lines + SPP proxy):
  - k// = n_air k0
  - k// = n_substrate k0
  - k// = n_EML k0
  - k// = n_WGP k0 (max(Re(n)) among organic layers)
  - k_spp(λ) at ETL/cathode1: k0*sqrt( eps_ETL*eps_cathode1 / (eps_ETL+eps_cathode1) )

Python compatibility:
- Avoids PEP604 (T1|T2) and builtin generics (list[str]) for Python<3.10/<3.9.

Run (from repo root = directory containing out_gpvm_step2):
  python gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py

Cache:
- If *.npy exist, they are loaded and only figures are regenerated.
- Delete *.npy to force recompute.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent  # repo root

sys.path.insert(0, str(BASE))

from oled_cavity_phase3p1_policyB import build_current_base
from run_phase3_opt_then_gpvm_eml_profile import nm_to_m
from gpvm.kz import kz
from gpvm.system_matrix import stack_transfer_matrix, rt_from_transfer_matrix

DEFAULT_BEST_JSON = BASE / "out_gpvm_step2" / "best_geometry.json"
BEST_JSON_PATH = DEFAULT_BEST_JSON
OUT_DIR = HERE
TAG = ""

# wavelength scan control (defaults: centered on best_geometry['lambda_nm'])
LAM_CENTER_NM_ARG = None  # type: float | None
LAM_MIN_NM_ARG = None     # type: float | None
LAM_MAX_NM_ARG = None     # type: float | None
LAM_SPAN_NM = 300.0
N_LAM = 201


# -------------------------
# GPVM equations (page 9, Eq. 26-28) + Eq. 30
# -------------------------
def _D(rA: complex, rB: complex, kz_e: complex, d_eml_m: float) -> complex:
    return 1.0 - (rA * rB * np.exp(1j * 2.0 * kz_e * d_eml_m))


def Pe_TE_h_eq26(rA: complex, rB: complex, kz_e: complex, d_eml_m: float, z_ex_m: float) -> float:
    """Eq. (26): spectral power density per unit solid angle for TE, horizontal dipole."""
    RA = abs(rA) ** 2
    RB = abs(rB) ** 2
    D = _D(rA, rB, kz_e, d_eml_m)

    # NOTE: Eq.(26) has (1 - R_B) multiplying the r_A term, and (1 - R_A) multiplying the r_B term.
    num_A = 1.0 + rA * np.exp(1j * 2.0 * kz_e * z_ex_m)
    num_B = 1.0 + rB * np.exp(1j * 2.0 * kz_e * (d_eml_m - z_ex_m))

    term_A = (1.0 - RB) * (abs(num_A / D) ** 2)
    term_B = (1.0 - RA) * (abs(num_B / D) ** 2)

    return float(np.real((3.0 / (16.0 * np.pi)) * (term_A + term_B)))


def Pe_TM_h_eq27(rA: complex, rB: complex, kz_e: complex, d_eml_m: float, z_ex_m: float, u: float) -> float:
    """Eq. (27): spectral power density per unit solid angle for TM, horizontal dipole."""
    RA = abs(rA) ** 2
    RB = abs(rB) ** 2
    D = _D(rA, rB, kz_e, d_eml_m)

    num_A = 1.0 + rA * np.exp(1j * 2.0 * kz_e * z_ex_m)
    num_B = 1.0 + rB * np.exp(1j * 2.0 * kz_e * (d_eml_m - z_ex_m))

    term_A = (1.0 - RB) * (abs(num_A / D) ** 2)
    term_B = (1.0 - RA) * (abs(num_B / D) ** 2)

    return float(np.real((3.0 / (16.0 * np.pi)) * (1.0 - u ** 2) * (term_A + term_B)))


def Pe_TM_v_eq28(rA: complex, rB: complex, kz_e: complex, d_eml_m: float, z_ex_m: float, u: float) -> float:
    """Eq. (28): spectral power density per unit solid angle for TM, vertical dipole."""
    RA = abs(rA) ** 2
    RB = abs(rB) ** 2
    D = _D(rA, rB, kz_e, d_eml_m)

    # NOTE: Eq.(28) uses MINUS signs in the numerators.
    num_A = 1.0 - rA * np.exp(1j * 2.0 * kz_e * z_ex_m)
    num_B = 1.0 - rB * np.exp(1j * 2.0 * kz_e * (d_eml_m - z_ex_m))

    term_A = (1.0 - RB) * (abs(num_A / D) ** 2)
    term_B = (1.0 - RA) * (abs(num_B / D) ** 2)

    return float(np.real((3.0 / (8.0 * np.pi)) * (u ** 2) * (term_A + term_B)))


def K_from_Pe_eq30(Pe: float, u: float) -> float:
    """Eq. (30): K(u) = π/(1-u^2) * Pe(u)."""
    denom = max(1e-30, (1.0 - u ** 2))
    return float(np.real(np.pi / denom * Pe))


# -------------------------
# Helpers
# -------------------------
def fwhm_nm(lam_nm: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    lam_nm = np.asarray(lam_nm, float)
    y = np.asarray(y, float)
    if np.max(y) <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    i_peak = int(np.argmax(y))
    peak_nm = float(lam_nm[i_peak])
    y_norm = y / (np.max(y) + 1e-30)
    half = 0.5

    left_nm = None
    for i in range(i_peak, 0, -1):
        if y_norm[i] >= half and y_norm[i - 1] < half:
            x0, x1 = lam_nm[i - 1], lam_nm[i]
            y0, y1 = y_norm[i - 1], y_norm[i]
            t = (half - y0) / (y1 - y0 + 1e-30)
            left_nm = float(x0 + t * (x1 - x0))
            break

    right_nm = None
    for i in range(i_peak, len(lam_nm) - 1):
        if y_norm[i] >= half and y_norm[i + 1] < half:
            x0, x1 = lam_nm[i], lam_nm[i + 1]
            y0, y1 = y_norm[i], y_norm[i + 1]
            t = (half - y0) / (y1 - y0 + 1e-30)
            right_nm = float(x0 + t * (x1 - x0))
            break

    if left_nm is None or right_nm is None:
        return peak_nm, float("nan"), float("nan"), float("nan")

    return peak_nm, float(right_nm - left_nm), left_nm, right_nm


def desired_lam_grid_nm() -> 'np.ndarray':
    """Return wavelength grid (nm).

    Priority:
      1) explicit --lam-min-nm/--lam-max-nm (either/both)
      2) --lam-center-nm
      3) best_geometry.json['lambda_nm']
    Range defaults to center ± (LAM_SPAN_NM/2).
    """
    global LAM_CENTER_NM_ARG, LAM_MIN_NM_ARG, LAM_MAX_NM_ARG, LAM_SPAN_NM, N_LAM
    # center
    if LAM_CENTER_NM_ARG is not None:
        lam0 = float(LAM_CENTER_NM_ARG)
    else:
        best = json.loads(BEST_JSON_PATH.read_text(encoding='utf-8'))
        lam0 = float(best.get('lambda_nm', 650.0))
    span = float(LAM_SPAN_NM)
    lam_min = lam0 - 0.5 * span
    lam_max = lam0 + 0.5 * span
    if LAM_MIN_NM_ARG is not None:
        lam_min = float(LAM_MIN_NM_ARG)
    if LAM_MAX_NM_ARG is not None:
        lam_max = float(LAM_MAX_NM_ARG)
    # sanitize
    if not (lam_max > lam_min):
        lam_max = lam_min + 1.0
    if lam_min <= 1.0:
        lam_min = 1.0
    n = int(N_LAM)
    if n < 3:
        n = 3
    return np.linspace(lam_min, lam_max, n)


def build_n0_with_ito_as_metal() -> Dict:
    """Build optical constants (constant n) and apply ITO -> (0.14 + 2000 i)."""
    n0, _ = build_current_base()
    n0 = dict(n0)
    n0["ITO"] = 0.14 + 1j * 2000.0
    return n0


def region_lines_kpar_um(
    lam_nm: np.ndarray, n0: Dict, organic_keys: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """Boundary lines in k// [1/µm] for plotting."""
    lam_um = lam_nm * 1e-3  # nm -> µm
    k0_um = 2.0 * np.pi / np.maximum(lam_um, 1e-30)

    n_air = float(np.real(n0.get("air", 1.0)))
    n_sub = float(np.real(n0.get("substrate", 1.52)))
    n_eml = float(np.real(n0.get("EML", 2.0)))

    if organic_keys is None:
        organic_keys = ["CPL", "ETL", "EML", "EBL", "HTL", "Rprime", "pHTL"]
    n_wg = max(float(np.real(n0[k])) for k in organic_keys if k in n0)

    # SPP proxy at ETL/cathode1 interface
    eps_d = n0["ETL"] ** 2
    eps_m = n0["cathode1"] ** 2
    kspp_um = np.real(k0_um * np.sqrt(eps_d * eps_m / (eps_d + eps_m)))

    return {
        "air": n_air * k0_um,
        "substrate": n_sub * k0_um,
        "EML": n_eml * k0_um,
        "WGP": n_wg * k0_um,
        "SPP(ETL/cathode1)": kspp_um,
    }


def overlay_region_lines_lambda_x(ax: plt.Axes, lam_nm: np.ndarray, kpar_grid_um: np.ndarray, n0: Dict) -> None:
    lines = region_lines_kpar_um(lam_nm, n0)
    # solid lines
    for key in ["air", "substrate", "EML", "WGP"]:
        ax.plot(lam_nm, lines[key], "w-", lw=1.2)
    # SPP as dashed
    ax.plot(lam_nm, lines["SPP(ETL/cathode1)"], "w--", lw=1.2)

    # avoid clipping
    y_max = float(np.nanmax(np.vstack([v for v in lines.values()])))
    ax.set_ylim(kpar_grid_um[0], max(kpar_grid_um[-1], 1.01 * y_max))


def overlay_region_lines_kpar_x(ax: plt.Axes, lam_nm: np.ndarray, kpar_grid_um: np.ndarray, n0: Dict) -> None:
    lines = region_lines_kpar_um(lam_nm, n0)
    for key in ["air", "substrate", "EML", "WGP"]:
        ax.plot(lines[key], lam_nm, "w-", lw=1.2)
    ax.plot(lines["SPP(ETL/cathode1)"], lam_nm, "w--", lw=1.2)

    x_max = float(np.nanmax(np.vstack([v for v in lines.values()])))
    ax.set_xlim(kpar_grid_um[0], max(kpar_grid_um[-1], 1.01 * x_max))


# -------------------------
# Core computation
# -------------------------
def compute_arrays() -> Dict[str, np.ndarray]:
    best = json.loads(BEST_JSON_PATH.read_text(encoding="utf-8"))
    d_best_nm = {k: float(v) for k, v in best["d_best_nm"].items()}

    n0 = build_n0_with_ito_as_metal()
    n_e = complex(float(np.real(n0["EML"])), 0.0)

    d_by_name_m = {k: nm_to_m(v) for k, v in d_best_nm.items()}
    d_eml_m = nm_to_m(d_best_nm["EML"])
    z_ex_m = nm_to_m(0.5 * d_best_nm["EML"])  # strict: EML center

    # Left stack: anode | ITO | pHTL | Rprime | HTL | EBL | EML
    n_left = [n0["anode"], n0["ITO"], n0["pHTL"], n0["Rprime"], n0["HTL"], n0["EBL"], n_e]
    d_left = [
        d_by_name_m["ITO"],
        d_by_name_m["pHTL"],
        d_by_name_m["Rprime"],
        d_by_name_m["HTL"],
        d_by_name_m["EBL"],
    ]

    # Right stack: EML | ETL | cathode1 | cathode | CPL | air
    n_right = [n_e, n0["ETL"], n0["cathode1"], n0["cathode"], n0["CPL"], n0["air"]]
    d_right = [
        d_by_name_m["ETL"],
        d_by_name_m["cathode1"],
        d_by_name_m["cathode"],
        d_by_name_m["CPL"],
    ]

    # λ grid
    lam_nm = desired_lam_grid_nm()
    lam_m = lam_nm * 1e-9

    # k// grid independent of λ
    u_max = 0.99
    k0_max = 2.0 * np.pi / float(np.min(lam_m))
    kpar_max = u_max * float(np.real(n_e)) * k0_max
    kpar_grid = np.linspace(0.0, kpar_max, 240)  # [1/m]
    kpar_grid_um = kpar_grid * 1e-6

    shp = (lam_nm.size, kpar_grid.size)
    K_TE_h = np.full(shp, np.nan, dtype=float)
    K_TM_h = np.full(shp, np.nan, dtype=float)
    K_TM_v = np.full(shp, np.nan, dtype=float)

    # u=0 spectra (for validation)
    K_TE_h_u0 = np.zeros_like(lam_nm, dtype=float)
    K_TM_h_u0 = np.zeros_like(lam_nm, dtype=float)
    K_TM_v_u0 = np.zeros_like(lam_nm, dtype=float)

    for i_lam, lm in enumerate(lam_m):
        k0 = 2.0 * np.pi / float(lm)

        # u=0 rA/rB for TE & TM
        u0 = 0.0

        # TE
        M_left_te = stack_transfer_matrix(n_left, d_left, n_e=n_e, u=u0, lambda0_m=float(lm), pol="TE")
        rA_te = -M_left_te[0, 1] / M_left_te[0, 0]
        M_right_te = stack_transfer_matrix(n_right, d_right, n_e=n_e, u=u0, lambda0_m=float(lm), pol="TE")
        rB_te, _ = rt_from_transfer_matrix(M_right_te)

        kz_e = kz(n_e, n_e, u0, float(lm))
        Pe_te = Pe_TE_h_eq26(rA_te, rB_te, kz_e, d_eml_m, z_ex_m)
        K_TE_h_u0[i_lam] = K_from_Pe_eq30(Pe_te, u0)

        # TM
        M_left_tm = stack_transfer_matrix(n_left, d_left, n_e=n_e, u=u0, lambda0_m=float(lm), pol="TM")
        rA_tm = -M_left_tm[0, 1] / M_left_tm[0, 0]
        M_right_tm = stack_transfer_matrix(n_right, d_right, n_e=n_e, u=u0, lambda0_m=float(lm), pol="TM")
        rB_tm, _ = rt_from_transfer_matrix(M_right_tm)

        Pe_tmh = Pe_TM_h_eq27(rA_tm, rB_tm, kz_e, d_eml_m, z_ex_m, u0)
        Pe_tmv = Pe_TM_v_eq28(rA_tm, rB_tm, kz_e, d_eml_m, z_ex_m, u0)

        K_TM_h_u0[i_lam] = K_from_Pe_eq30(Pe_tmh, u0)
        K_TM_v_u0[i_lam] = K_from_Pe_eq30(Pe_tmv, u0)  # =0 at u=0

        # sweep k//
        for j_k, kpar in enumerate(kpar_grid):
            u = float(kpar / (float(np.real(n_e)) * k0 + 1e-30))
            if u < 0.0 or u > u_max:
                continue

            kz_e_u = kz(n_e, n_e, u, float(lm))

            # TE
            M_left_te = stack_transfer_matrix(n_left, d_left, n_e=n_e, u=u, lambda0_m=float(lm), pol="TE")
            rA_te = -M_left_te[0, 1] / M_left_te[0, 0]
            M_right_te = stack_transfer_matrix(n_right, d_right, n_e=n_e, u=u, lambda0_m=float(lm), pol="TE")
            rB_te, _ = rt_from_transfer_matrix(M_right_te)
            Pe_te = Pe_TE_h_eq26(rA_te, rB_te, kz_e_u, d_eml_m, z_ex_m)
            K_TE_h[i_lam, j_k] = K_from_Pe_eq30(Pe_te, u)

            # TM
            M_left_tm = stack_transfer_matrix(n_left, d_left, n_e=n_e, u=u, lambda0_m=float(lm), pol="TM")
            rA_tm = -M_left_tm[0, 1] / M_left_tm[0, 0]
            M_right_tm = stack_transfer_matrix(n_right, d_right, n_e=n_e, u=u, lambda0_m=float(lm), pol="TM")
            rB_tm, _ = rt_from_transfer_matrix(M_right_tm)

            Pe_tmh = Pe_TM_h_eq27(rA_tm, rB_tm, kz_e_u, d_eml_m, z_ex_m, u)
            Pe_tmv = Pe_TM_v_eq28(rA_tm, rB_tm, kz_e_u, d_eml_m, z_ex_m, u)

            K_TM_h[i_lam, j_k] = K_from_Pe_eq30(Pe_tmh, u)
            K_TM_v[i_lam, j_k] = K_from_Pe_eq30(Pe_tmv, u)

    # derived maps
    K_TE_only = (2.0 / 3.0) * K_TE_h
    K_TM_h_only = (2.0 / 3.0) * K_TM_h
    K_TM_v_only = (1.0 / 3.0) * K_TM_v
    K_iso = K_TE_only + K_TM_h_only + K_TM_v_only

    # u=0 derived
    K_TE_only_u0 = (2.0 / 3.0) * K_TE_h_u0
    K_TM_h_only_u0 = (2.0 / 3.0) * K_TM_h_u0
    K_TM_v_only_u0 = (1.0 / 3.0) * K_TM_v_u0
    K_iso_u0 = K_TE_only_u0 + K_TM_h_only_u0 + K_TM_v_only_u0

    # save cache
    np.save(OUT_DIR / "lam_nm.npy", lam_nm)
    np.save(OUT_DIR / "kpar_grid_um.npy", kpar_grid_um)
    np.save(OUT_DIR / "K_TE_h_map.npy", K_TE_h)
    np.save(OUT_DIR / "K_TM_h_map.npy", K_TM_h)
    np.save(OUT_DIR / "K_TM_v_map.npy", K_TM_v)
    np.save(OUT_DIR / "K_TE_only_map.npy", K_TE_only)
    np.save(OUT_DIR / "K_TM_h_only_map.npy", K_TM_h_only)
    np.save(OUT_DIR / "K_TM_v_only_map.npy", K_TM_v_only)
    np.save(OUT_DIR / "K_iso_map.npy", K_iso)

    np.save(OUT_DIR / "K_TE_h_u0.npy", K_TE_h_u0)
    np.save(OUT_DIR / "K_TM_h_u0.npy", K_TM_h_u0)
    np.save(OUT_DIR / "K_TM_v_u0.npy", K_TM_v_u0)

    np.save(OUT_DIR / "K_TE_only_u0.npy", K_TE_only_u0)
    np.save(OUT_DIR / "K_TM_h_only_u0.npy", K_TM_h_only_u0)
    np.save(OUT_DIR / "K_TM_v_only_u0.npy", K_TM_v_only_u0)
    np.save(OUT_DIR / "K_iso_u0.npy", K_iso_u0)

    return {
        'lam_nm': lam_nm,
        'kpar_grid_um': kpar_grid_um,
        'K_TE_h': K_TE_h,
        'K_TM_h': K_TM_h,
        'K_TM_v': K_TM_v,
        'K_TE_only': K_TE_only,
        'K_TM_h_only': K_TM_h_only,
        'K_TM_v_only': K_TM_v_only,
        'K_iso': K_iso,
        'K_TE_h_u0': K_TE_h_u0,
        'K_TM_h_u0': K_TM_h_u0,
        'K_TM_v_u0': K_TM_v_u0,
        'K_TE_only_u0': K_TE_only_u0,
        'K_TM_h_only_u0': K_TM_h_only_u0,
        'K_TM_v_only_u0': K_TM_v_only_u0,
        'K_iso_u0': K_iso_u0,
    }



def compute_u0_only(lam_nm: np.ndarray) -> Dict[str, np.ndarray]:
    """Recompute u=0 spectra for all components (fast), without recomputing full maps."""
    best = json.loads(BEST_JSON_PATH.read_text(encoding="utf-8"))
    d_best_nm = {k: float(v) for k, v in best["d_best_nm"].items()}

    n0 = build_n0_with_ito_as_metal()
    n_e = complex(float(np.real(n0["EML"])), 0.0)
    d_by_name_m = {k: nm_to_m(v) for k, v in d_best_nm.items()}

    d_eml_m = nm_to_m(d_best_nm["EML"])
    z_ex_m = nm_to_m(0.5 * d_best_nm["EML"])  # strict: EML center

    n_left = [n0["anode"], n0["ITO"], n0["pHTL"], n0["Rprime"], n0["HTL"], n0["EBL"], n_e]
    d_left = [d_by_name_m["ITO"], d_by_name_m["pHTL"], d_by_name_m["Rprime"], d_by_name_m["HTL"], d_by_name_m["EBL"]]

    n_right = [n_e, n0["ETL"], n0["cathode1"], n0["cathode"], n0["CPL"], n0["air"]]
    d_right = [d_by_name_m["ETL"], d_by_name_m["cathode1"], d_by_name_m["cathode"], d_by_name_m["CPL"]]

    lam_m = lam_nm * 1e-9
    u0 = 0.0

    K_TE_only_u0 = np.zeros_like(lam_nm, dtype=float)
    K_TM_h_only_u0 = np.zeros_like(lam_nm, dtype=float)
    K_TM_v_only_u0 = np.zeros_like(lam_nm, dtype=float)
    K_TE_only_u0 = np.zeros_like(lam_nm, dtype=float)
    K_TM_h_only_u0 = np.zeros_like(lam_nm, dtype=float)
    K_TM_v_only_u0 = np.zeros_like(lam_nm, dtype=float)
    K_iso_u0 = np.zeros_like(lam_nm, dtype=float)

    # also store raw components if needed later
    K_TE_h_u0 = np.zeros_like(lam_nm, dtype=float)
    K_TM_h_u0 = np.zeros_like(lam_nm, dtype=float)
    K_TM_v_u0 = np.zeros_like(lam_nm, dtype=float)

    for i_lam, lm in enumerate(lam_m):
        M_left = stack_transfer_matrix(n_left, d_left, n_e=n_e, u=u0, lambda0_m=float(lm), pol="TE")
        rA_te = -M_left[0, 1] / M_left[0, 0]
        M_right = stack_transfer_matrix(n_right, d_right, n_e=n_e, u=u0, lambda0_m=float(lm), pol="TE")
        rB_te, _ = rt_from_transfer_matrix(M_right)
        kz_e = kz(n_e, n_e, u0, float(lm))

        Pe_te_h = Pe_TE_h_eq26(rA_te, rB_te, kz_e, d_eml_m, z_ex_m)
        K_te_h = K_from_Pe_eq30(Pe_te_h, u0)

        # TM uses its own rA/rB at the same interfaces (but pol="TM")
        M_left_tm = stack_transfer_matrix(n_left, d_left, n_e=n_e, u=u0, lambda0_m=float(lm), pol="TM")
        rA_tm = -M_left_tm[0, 1] / M_left_tm[0, 0]
        M_right_tm = stack_transfer_matrix(n_right, d_right, n_e=n_e, u=u0, lambda0_m=float(lm), pol="TM")
        rB_tm, _ = rt_from_transfer_matrix(M_right_tm)

        Pe_tm_h = Pe_TM_h_eq27(rA_tm, rB_tm, kz_e, d_eml_m, z_ex_m, u0)
        Pe_tm_v = Pe_TM_v_eq28(rA_tm, rB_tm, kz_e, d_eml_m, z_ex_m, u0)
        K_tm_h = K_from_Pe_eq30(Pe_tm_h, u0)
        K_tm_v = K_from_Pe_eq30(Pe_tm_v, u0)

        K_TE_h_u0[i_lam] = K_te_h
        K_TM_h_u0[i_lam] = K_tm_h
        K_TM_v_u0[i_lam] = K_tm_v

        K_TE_only_u0[i_lam] = (2.0 / 3.0) * K_te_h
        K_TM_h_only_u0[i_lam] = (2.0 / 3.0) * K_tm_h
        K_TM_v_only_u0[i_lam] = (1.0 / 3.0) * K_tm_v
        K_iso_u0[i_lam] = K_TE_only_u0[i_lam] + K_TM_h_only_u0[i_lam] + K_TM_v_only_u0[i_lam]

    return {
        "K_TE_h_u0": K_TE_h_u0,
        "K_TM_h_u0": K_TM_h_u0,
        "K_TM_v_u0": K_TM_v_u0,
        "K_TE_only_u0": K_TE_only_u0,
        "K_TM_h_only_u0": K_TM_h_only_u0,
        "K_TM_v_only_u0": K_TM_v_only_u0,
        "K_iso_u0": K_iso_u0,
    }

def load_or_compute() -> Dict[str, np.ndarray]:
    """Load cached maps if present; (re)compute u=0 spectra if missing; otherwise compute everything."""
    map_files = [
        "lam_nm.npy",
        "kpar_grid_um.npy",
        "K_TE_only_map.npy",
        "K_TM_h_only_map.npy",
        "K_TM_v_only_map.npy",
        "K_iso_map.npy",
        "K_TE_h_map.npy",
        "K_TM_h_map.npy",
        "K_TM_v_map.npy",
    ]
    have_maps = all((OUT_DIR / f).exists() for f in map_files)

    # if cached wavelength grid does not match current requested grid, ignore cache
    if have_maps:
        desired_lam = desired_lam_grid_nm()
        lam_cached = np.load(OUT_DIR / 'lam_nm.npy')
        if (lam_cached.size != desired_lam.size) or (abs(float(lam_cached[0]) - float(desired_lam[0])) > 1e-9) or (abs(float(lam_cached[-1]) - float(desired_lam[-1])) > 1e-9):
            print('Cache wavelength grid mismatch; recomputing maps.')
            return compute_arrays()

    if have_maps:
        out = {
            "lam_nm": np.load(OUT_DIR / "lam_nm.npy"),
            "kpar_grid_um": np.load(OUT_DIR / "kpar_grid_um.npy"),
            "K_TE_h": np.load(OUT_DIR / "K_TE_h_map.npy"),
            "K_TM_h": np.load(OUT_DIR / "K_TM_h_map.npy"),
            "K_TM_v": np.load(OUT_DIR / "K_TM_v_map.npy"),
            "K_TE_only": np.load(OUT_DIR / "K_TE_only_map.npy"),
            "K_TM_h_only": np.load(OUT_DIR / "K_TM_h_only_map.npy"),
            "K_TM_v_only": np.load(OUT_DIR / "K_TM_v_only_map.npy"),
            "K_iso": np.load(OUT_DIR / "K_iso_map.npy"),
        }

        u0_files = [
            "K_TE_h_u0.npy", "K_TM_h_u0.npy", "K_TM_v_u0.npy",
            "K_TE_only_u0.npy", "K_TM_h_only_u0.npy", "K_TM_v_only_u0.npy",
            "K_iso_u0.npy",
        ]
        if all((OUT_DIR / f).exists() for f in u0_files):
            out.update({
                "K_TE_h_u0": np.load(OUT_DIR / "K_TE_h_u0.npy"),
                "K_TM_h_u0": np.load(OUT_DIR / "K_TM_h_u0.npy"),
                "K_TM_v_u0": np.load(OUT_DIR / "K_TM_v_u0.npy"),
                "K_TE_only_u0": np.load(OUT_DIR / "K_TE_only_u0.npy"),
                "K_TM_h_only_u0": np.load(OUT_DIR / "K_TM_h_only_u0.npy"),
                "K_TM_v_only_u0": np.load(OUT_DIR / "K_TM_v_only_u0.npy"),
                "K_iso_u0": np.load(OUT_DIR / "K_iso_u0.npy"),
            })
        else:
            u0 = compute_u0_only(out["lam_nm"])
            # save for next time
            np.save(OUT_DIR / "K_TE_h_u0.npy", u0["K_TE_h_u0"])
            np.save(OUT_DIR / "K_TM_h_u0.npy", u0["K_TM_h_u0"])
            np.save(OUT_DIR / "K_TM_v_u0.npy", u0["K_TM_v_u0"])
            np.save(OUT_DIR / "K_TE_only_u0.npy", u0["K_TE_only_u0"])
            np.save(OUT_DIR / "K_TM_h_only_u0.npy", u0["K_TM_h_only_u0"])
            np.save(OUT_DIR / "K_TM_v_only_u0.npy", u0["K_TM_v_only_u0"])
            np.save(OUT_DIR / "K_iso_u0.npy", u0["K_iso_u0"])
            out.update(u0)
        return out

    return compute_arrays()


# -------------------------
# Plotting
# -------------------------
def _heatmap_lambda_x_kpar_y(
    lam_nm: np.ndarray,
    kpar_grid_um: np.ndarray,
    Z_lam_kpar: np.ndarray,
    title: str,
    out_png: Path,
    log10: bool = False,
) -> None:
    n0 = build_n0_with_ito_as_metal()

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    data = Z_lam_kpar.T
    if log10:
        data = np.log10(np.maximum(data, 1e-12))
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[lam_nm[0], lam_nm[-1], kpar_grid_um[0], kpar_grid_um[-1]],
    )
    overlay_region_lines_lambda_x(ax, lam_nm, kpar_grid_um, n0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$k_{//}$ (1/µm)")
    ax.set_title(title + (" (log10)" if log10 else ""))
    fig.colorbar(im, ax=ax, label=("log10" if log10 else "") + "K (a.u.)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _heatmap_kpar_x_lambda_y(
    lam_nm: np.ndarray,
    kpar_grid_um: np.ndarray,
    Z_lam_kpar: np.ndarray,
    title: str,
    out_png: Path,
    log10: bool = False,
) -> None:
    n0 = build_n0_with_ito_as_metal()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    data = Z_lam_kpar
    if log10:
        data = np.log10(np.maximum(data, 1e-12))
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[kpar_grid_um[0], kpar_grid_um[-1], lam_nm[0], lam_nm[-1]],
    )
    overlay_region_lines_kpar_x(ax, lam_nm, kpar_grid_um, n0)
    ax.set_xlabel(r"$k_{//}$ (1/µm)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title(title + (" (log10)" if log10 else ""))
    fig.colorbar(im, ax=ax, label=("log10" if log10 else "") + "K (a.u.)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def make_figures(data: Dict[str, np.ndarray]) -> None:
    lam_nm = data["lam_nm"]
    kpar_grid_um = data["kpar_grid_um"]

    items = [
        ("K_iso", data["K_iso"]),
        ("K_TE_only", data["K_TE_only"]),
        ("K_TM_h_only", data["K_TM_h_only"]),
        ("K_TM_v_only", data["K_TM_v_only"]),
    ]

    for name, Z in items:
        _heatmap_lambda_x_kpar_y(
            lam_nm, kpar_grid_um, Z, f"Strict-EML GPVM {name}(λ,k//)", OUT_DIR / f"heatmap_lambda_x_kpar_y__{name}__linear{TAG}.png", log10=False
        )
        _heatmap_lambda_x_kpar_y(
            lam_nm, kpar_grid_um, Z, f"Strict-EML GPVM {name}(λ,k//)", OUT_DIR / f"heatmap_lambda_x_kpar_y__{name}__log{TAG}.png", log10=True
        )
        _heatmap_kpar_x_lambda_y(
            lam_nm, kpar_grid_um, Z, f"Strict-EML GPVM {name}(λ,k//)", OUT_DIR / f"heatmap_kpar_x_lambda_y__{name}__linear{TAG}.png", log10=False
        )
        _heatmap_kpar_x_lambda_y(
            lam_nm, kpar_grid_um, Z, f"Strict-EML GPVM {name}(λ,k//)", OUT_DIR / f"heatmap_kpar_x_lambda_y__{name}__log{TAG}.png", log10=True
        )
    # k//=0 slice checks (for each reported quantity)
    check_items = [
        ("K_iso", data["K_iso"], data["K_iso_u0"], "Normalized K_iso"),
        ("K_TE_only", data["K_TE_only"], data["K_TE_only_u0"], "Normalized K_TE_only"),
        ("K_TM_h_only", data["K_TM_h_only"], data["K_TM_h_only_u0"], "Normalized K_TM_h_only"),
        ("K_TM_v_only", data["K_TM_v_only"], data["K_TM_v_only_u0"], "Normalized K_TM_v_only"),
    ]

    for name, Z, Ku0, ylabel in check_items:
        K_k0 = Z[:, 0]
        diff = float(np.nanmax(np.abs(K_k0 - Ku0)))

        peak_nm, fwhm, left_nm, right_nm = fwhm_nm(lam_nm, K_k0)

        yA = K_k0 / (np.nanmax(K_k0) + 1e-30)
        yB = Ku0 / (np.nanmax(Ku0) + 1e-30)

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.plot(lam_nm, yA, label=r"k//=0 slice from heatmap")
        ax.plot(lam_nm, yB, "--", label=r"recomputed u=0")
        ax.axvline(peak_nm, linestyle=":", linewidth=1)
        if np.isfinite(left_nm) and np.isfinite(right_nm):
            ax.axvline(left_nm, linestyle=":", linewidth=1)
            ax.axvline(right_nm, linestyle=":", linewidth=1)
            ax.axhline(0.5, linestyle=":", linewidth=1)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(ylabel)
        ax.set_title("k//=0 check ({}): max|Δ|={:.3e}\npeak={:.1f} nm, FWHM={:.2f} nm".format(name, diff, peak_nm, fwhm))
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"check_k0_matches_u0_peak_fwhm__{name}{TAG}.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(lam_nm, yA - yB)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("(heatmap k0) - (recomputed u0)")
        ax.set_title("Difference ({}), should be ~0 | max|Δ|={:.3e}".format(name, diff))
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"check_u0_vs_k0__{name}{TAG}.png", dpi=220)
        plt.close(fig)

        print("k//=0 check ({}): max|Δ|={:.3e}, peak={:.1f} nm, FWHM={:.2f} nm".format(name, diff, peak_nm, fwhm))

    # Combined u=0 slices (normalized) for quick inspection
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, Z, Ku0, _ in check_items:
        y = Ku0 / (np.nanmax(Ku0) + 1e-30)
        ax.plot(lam_nm, y, label=name)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized (u=0)")
    ax.set_title("Strict-EML u=0 spectra (normalized): K_iso & components")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"slice_u0_components__normalized{TAG}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    global BEST_JSON_PATH, OUT_DIR, TAG

    ap = argparse.ArgumentParser(description="Strict-EML GPVM: K(lambda,k//) heatmaps")
    ap.add_argument("--best-json", type=str, default=str(DEFAULT_BEST_JSON),
                    help="Path to best geometry JSON (default: ./out_gpvm_step2/best_geometry.json)")
    ap.add_argument("--outdir", type=str, default=str(HERE),
                    help="Directory for cache (*.npy) and figures (default: this script directory)")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional suffix for figure filenames only, e.g. _R/_G/_B (cache names unchanged)")
    ap.add_argument("--lam-center-nm", type=float, default=None,
                    help="Center wavelength (nm) for scans; default: best_geometry['lambda_nm']")
    ap.add_argument("--lam-span-nm", type=float, default=300.0,
                    help="Scan span (nm). Range = center ± span/2 when min/max not fully specified (default: 300)")
    ap.add_argument("--lam-min-nm", type=float, default=None,
                    help="Override scan minimum wavelength (nm)")
    ap.add_argument("--lam-max-nm", type=float, default=None,
                    help="Override scan maximum wavelength (nm)")
    ap.add_argument("--n-lam", type=int, default=201,
                    help="Number of wavelength samples (default: 201)")
    args = ap.parse_args()

    best_json_in = Path(args.best_json).expanduser()
    BEST_JSON_PATH = best_json_in if best_json_in.is_absolute() else (BASE / best_json_in)

    out_dir_in = Path(args.outdir).expanduser()
    OUT_DIR = out_dir_in if out_dir_in.is_absolute() else (BASE / out_dir_in)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    TAG = str(args.tag)

    # set wavelength scan globals
    global LAM_CENTER_NM_ARG, LAM_MIN_NM_ARG, LAM_MAX_NM_ARG, LAM_SPAN_NM, N_LAM
    LAM_CENTER_NM_ARG = args.lam_center_nm
    LAM_MIN_NM_ARG = args.lam_min_nm
    LAM_MAX_NM_ARG = args.lam_max_nm
    LAM_SPAN_NM = float(args.lam_span_nm)
    N_LAM = int(args.n_lam)

    data = load_or_compute()
    make_figures(data)


if __name__ == "__main__":
    main()
