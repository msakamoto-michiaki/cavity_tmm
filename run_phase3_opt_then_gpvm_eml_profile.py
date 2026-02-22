#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(2) Phase3 geometry optimization (via existing phase3p1 code) -> GPVM |field|^2 profiles.

What this script does
---------------------
1) Uses phase3p1's existing optimization path (F(lambda) proxy) to select
   ETL thickness and the scale factor `s` for (HTL, Rprime).
2) With the best geometry, evaluates two "GPVM" style profiles at u=0 and
   lambda given by --lambda-nm (default: 650nm):

   (A) phase3-compatible cavity-GPVM (recommended for step (2)):
       Uses the *same* terminal reflections (B/T) and the same optical
       coordinate (zopt, LBT) as phase3's Eq.(54)/(66) proxy. This makes the
       resonance denominator identical to the phase3 F(lambda) objective.

   (B) strict-EML GPVM (physical EML boundaries):
       Uses rA/rB extracted at the physical EML interfaces and d_e = d_EML.
       This is physically stricter for the thin EML, but it does NOT share the
       same Fabry-Perot denominator as phase3's BT proxy unless BT is chosen at
       the EML interfaces.

Notes
-----
- In the GPVM paper / our gpvm TMM convention:
    TE uses tangential electric field amplitude.
    TM uses tangential magnetic field amplitude.
  Therefore this script outputs |E|^2 for TE and |H|^2 for TM.
- Why do we include both?
    Your phase3 optimization enforces the standing-wave peak in the EML via the
    BT-proxy (Eq.66 from rb_B/rt_T and zopt). If we compute strict-EML GPVM with
    rA/rB at the *physical* EML boundaries, the resonance condition generally
    shifts (different effective cavity), and the z-profile at lambda can
    look inconsistent. Profile (A) is the correct "consistency check" before
    moving on to full GPVM Fig.3/4/7/8 reproduction.

Outputs
-------
Writes to the directory containing --best-json (default: ./out_gpvm_step2/):
  - best_geometry*.json
  - z_eml_nm.npy, I_TE_h.npy, I_TM_h.npy, I_TM_v.npy, I_iso.npy
  - gpvm_eml_profiles_phase3BT.png   (recommended)
  - gpvm_eml_profiles_strictEML.png  (diagnostic)
"""

from __future__ import annotations

import json
from pathlib import Path

import argparse
import numpy as np
import matplotlib.pyplot as plt
from common.units import nm_to_m


#def nm_to_m(x_nm: float) -> float:
#    return float(x_nm) * 1e-9


def build_full_stack_from_phase3(n0: dict, d: dict):
    """Return (n_list, d_list_m, eml_index, z_ex_m).

    Full stack order (left->right): substrate | anode | ITO | pHTL | Rprime | HTL | EBL | EML | ETL | cathode1 | cathode | CPL | air
    """
    n_list = [
        n0["substrate"],
        n0["anode"],
        n0["ITO"],
        n0["pHTL"],
        n0["Rprime"],
        n0["HTL"],
        n0["EBL"],
        n0["EML"],
        n0["ETL"],
        n0["cathode1"],
        n0["cathode"],
        n0["CPL"],
        n0["air"],
    ]
    d_list_nm = [
        d["anode"],
        d["ITO"],
        d["pHTL"],
        d["Rprime"],
        d["HTL"],
        d["EBL"],
        d["EML"],
        d["ETL"],
        d["cathode1"],
        d["cathode"],
        d["CPL"],
    ]
    d_list_m = [nm_to_m(x) for x in d_list_nm]
    eml_index = 7
    z_ex_m = nm_to_m(float(d["EML"]) * 0.5)
    return n_list, d_list_m, eml_index, z_ex_m


def eml_profile_from_source_fields(
    z_eml_m: np.ndarray,
    z_ex_m: float,
    kz_e: complex,
    Ea_plus: complex,
    Ea_minus: complex,
    Eb_plus: complex,
    Eb_minus: complex,
) -> np.ndarray:
    """Return |field|^2(z) within EML given the source-plane fields."""
    z = np.asarray(z_eml_m, float)
    out = np.zeros_like(z, dtype=float)
    for i, zz in enumerate(z):
        if zz <= z_ex_m:
            dz = zz - z_ex_m
            f = (Ea_plus * np.exp(1j * kz_e * dz)) + (Ea_minus * np.exp(-1j * kz_e * dz))
        else:
            dz = zz - z_ex_m
            f = (Eb_plus * np.exp(1j * kz_e * dz)) + (Eb_minus * np.exp(-1j * kz_e * dz))
        out[i] = float(np.abs(f) ** 2)
    return out


def lcav_profile_internal_from_fields(
    *,
    fields,
    n_layers: dict,
    d_layers_nm: dict,
    z_ex_m: float,
    lambda0_m: float,
    u: float,
    pol: str,
    layer_order: list[str],
    n_e: complex,
    npts: int = 7001,
):
    """Return (z_nm, E2_norm, E0_amp_norm) across the *internal* LCAV stack.

    This reproduces the LCAV profile construction used in
    `run_bottom_metal_PEC_proxy.py`, but it does **not** define a bottom-metal
    ambient. Instead, it uses the already-solved source-plane fields (strict-EML
    GPVM) and propagates them across the internal cavity layers only.
    """
    from gpvm.kz import kz as kz_layer
    from gpvm.matrices import interface_matrix, layer_matrix

    # thickness bounds (nm)
    th_nm = [float(d_layers_nm[name]) for name in layer_order]
    bounds = np.cumsum([0.0] + th_nm)
    z_nm = np.linspace(0.0, bounds[-1], int(npts))

    # Left-side: source plane -> EML left boundary
    v_left = {}
    v_eml_left = layer_matrix(n_e, n_e, u, lambda0_m, float(z_ex_m)) @ np.array([fields.Ea_plus, fields.Ea_minus], dtype=complex)
    v_left["EML"] = v_eml_left

    prev = "EML"
    v_prev_left = v_eml_left
    for name in ["EBL", "HTL", "Rprime", "pHTL"]:
        I = interface_matrix(
            complex(n_layers[name]),
            complex(n_layers[prev] if prev != "EML" else n_e),
            n_e,
            u,
            lambda0_m,
            pol,
        )
        v_right = I @ v_prev_left
        L = layer_matrix(complex(n_layers[name]), n_e, u, lambda0_m, float(d_layers_nm[name]) * 1e-9)
        v_left[name] = L @ v_right
        prev = name
        v_prev_left = v_left[name]

    # Right-side: source plane -> EML right boundary -> ETL left boundary
    v_right_side = {}
    d_eml_m = float(d_layers_nm["EML"]) * 1e-9
    Lseg = layer_matrix(n_e, n_e, u, lambda0_m, (d_eml_m - float(z_ex_m)))
    v_eml_right = np.linalg.solve(Lseg, np.array([fields.Eb_plus, fields.Eb_minus], dtype=complex))
    v_right_side["EML"] = np.array([fields.Eb_plus, fields.Eb_minus], dtype=complex)

    I_eml_etl = interface_matrix(n_e, complex(n_layers["ETL"]), n_e, u, lambda0_m, pol)
    v_right_side["ETL"] = np.linalg.solve(I_eml_etl, v_eml_right)

    # Build complex field E(z)
    E = np.zeros_like(z_nm, dtype=complex)
    for i, name in enumerate(layer_order):
        z0, z1 = bounds[i], bounds[i + 1]
        mask = (z_nm >= z0) & (z_nm <= z1 + 1e-12)
        x_m = (z_nm[mask] - z0) * 1e-9
        kz_j = kz_layer(complex(n_layers[name]) if name != "EML" else n_e, n_e, u, lambda0_m)

        if name == "EML":
            z_ex_global = z0 + float(z_ex_m) * 1e9
            maskL = mask & (z_nm <= z_ex_global)
            maskR = mask & (z_nm >= z_ex_global)

            xL = (z_nm[maskL] - z0) * 1e-9
            EpL, EmL = v_left["EML"]
            E[maskL] = EpL * np.exp(1j * kz_j * xL) + EmL * np.exp(-1j * kz_j * xL)

            xR = (z_nm[maskR] - z_ex_global) * 1e-9
            EpR, EmR = v_right_side["EML"]
            E[maskR] = EpR * np.exp(1j * kz_j * xR) + EmR * np.exp(-1j * kz_j * xR)

        elif name in ["pHTL", "Rprime", "HTL", "EBL"]:
            Ep, Em = v_left[name]
            E[mask] = Ep * np.exp(1j * kz_j * x_m) + Em * np.exp(-1j * kz_j * x_m)

        else:  # ETL
            Ep, Em = v_right_side["ETL"]
            E[mask] = Ep * np.exp(1j * kz_j * x_m) + Em * np.exp(-1j * kz_j * x_m)

    E2 = np.abs(E) ** 2
    E2_norm = E2 / (np.max(E2) + 1e-30)
    E0_amp_norm = float(np.sqrt(E2_norm[0]))
    return z_nm, E2_norm, E0_amp_norm


def main():
    ap = argparse.ArgumentParser(description='Phase3 optimization then GPVM EML profile (u=0)')
    ap.add_argument('--lambda-nm', type=float, default=650.0,
                    help='Optimization/evaluation wavelength in nm (default: 650)')
    ap.add_argument('--best-json', type=str, default=str(Path(__file__).resolve().parent / 'out_gpvm_step2' / 'best_geometry.json'),
                    help='Output path for best geometry JSON (default: ./out_gpvm_step2/best_geometry.json)')
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    best_json_path_in = Path(args.best_json).expanduser()
    best_json_path = best_json_path_in if best_json_path_in.is_absolute() else (base / best_json_path_in)
    # Do not require the file to exist; create its parent dir.
    out_dir = best_json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import phase3 optimization
    from common.oled_cavity_phase3p1_policyB import build_current_base, optimize_etl_then_s, INTERNAL_ORDER
    from common.tmm_rewrap_utils_policyB import terminal_reflections_BT

    # GPVM modules
    from gpvm.kz import kz
    from gpvm.system_matrix import build_system_matrices_SA_SB, eq12_rt_from_SA_SB
    from gpvm.source_terms import source_terms
    from gpvm.source_plane import solve_source_plane_fields

    n0, d0 = build_current_base()

    # Match the default optimization settings used by phase3p1 main()
    cathode1_nm = 30.0
    lambda_target_nm = float(args.lambda_nm)
    pol_for_opt = 0  # TE

    df_coarse, best = optimize_etl_then_s(
        n0, d0,
        cathode1_nm=cathode1_nm,
        lambda_target=lambda_target_nm,
        pol=pol_for_opt,
    )

    (score, etl_best, s_best, d_best,
     z_phys, zopt, _E2z_S_abs, _E2z_BT_abs, _E2z_S_raw, _E2z_BT_raw,
     eml_start, eml_end, eml_center, LBT, phi_e) = best

    # --- GPVM evaluation at u=0, lambda=lambda_target_nm
    lam_m = float(lambda_target_nm) * 1e-9
    u_norm = 0.0
    n_e = float(np.real(n0["EML"]))

    # Phase3-compatible terminal reflections (B/T) at u=0
    rb_arr, rt_arr, _ = terminal_reflections_BT(
        n0=n0,
        d=d_best,
        lam_nm=np.array([float(lambda_target_nm)]),
        u=float(u_norm),
        pol=int(pol_for_opt),
        phi_b_mode="pi",
        tm_top_pi_shift=True,
        tm_bottom_pi_shift=False,
    )
    rb_used = complex(rb_arr[0])
    rt_used = complex(rt_arr[0])

    # --- (A) phase3-compatible cavity GPVM on optical coordinate
    # In phase3, propagation is expressed in optical coordinate zopt and cavity
    # optical length LBT (both in "nm * n"). The corresponding propagation
    # constant is k0 = 2pi/lambda (vacuum), and the reflection phase factors use
    # exp(i*2*k0*zopt) (same as beta=4pi/lambda in phase3 code).
    k0 = 2.0 * np.pi / lam_m
    d_cav_m = float(np.real(LBT)) * 1e-9
    # pick source position at the *physical* EML center as used by phase3 target
    idx_center = int(np.argmin(np.abs(z_phys - float(eml_center))))
    z_ex_opt_m = float(zopt[idx_center]) * 1e-9

    # BUGFIX / IMPORTANT:
    # `solve_source_plane_fields()` expects rA/rB referenced to the *source plane*.
    # The BT terminal reflections (rb_used, rt_used) are referenced at the cavity
    # terminals. Convert them using the same phase factors as phase3's ru/rd:
    #   ru(z) = rb_used * exp(i*2*k0*zopt),  rd(z) = rt_used * exp(i*2*k0*(LBT-zopt)).
    # At the source plane, zopt = z_ex_opt.
    rA_src_BT = rb_used * np.exp(1j * 2.0 * k0 * z_ex_opt_m)
    rB_src_BT = rt_used * np.exp(1j * 2.0 * k0 * (d_cav_m - z_ex_opt_m))

    # Build a z-grid in the EML region using the corresponding zopt values
    eml_mask = (z_phys >= float(eml_start)) & (z_phys <= float(eml_end))
    z_eml_phys_nm = z_phys[eml_mask]
    z_eml_opt_m = np.asarray(zopt[eml_mask], float) * 1e-9

    # TE: horizontal dipole only
    A_TE_h = source_terms(u_norm, pol="TE", orientation="h")
    f_TE_h_BT = solve_source_plane_fields(A_TE_h.A_plus, A_TE_h.A_minus, rA_src_BT, rB_src_BT, k0, d_cav_m, z_ex_opt_m)
    I_TE_h_BT = eml_profile_from_source_fields(
        z_eml_opt_m, z_ex_opt_m, k0,
        f_TE_h_BT.Ea_plus, f_TE_h_BT.Ea_minus, f_TE_h_BT.Eb_plus, f_TE_h_BT.Eb_minus,
    )

    # TM: horizontal and vertical
    A_TM_h = source_terms(u_norm, pol="TM", orientation="h")
    f_TM_h_BT = solve_source_plane_fields(A_TM_h.A_plus, A_TM_h.A_minus, rA_src_BT, rB_src_BT, k0, d_cav_m, z_ex_opt_m)
    I_TM_h_BT = eml_profile_from_source_fields(
        z_eml_opt_m, z_ex_opt_m, k0,
        f_TM_h_BT.Ea_plus, f_TM_h_BT.Ea_minus, f_TM_h_BT.Eb_plus, f_TM_h_BT.Eb_minus,
    )
    A_TM_v = source_terms(u_norm, pol="TM", orientation="v")
    f_TM_v_BT = solve_source_plane_fields(A_TM_v.A_plus, A_TM_v.A_minus, rA_src_BT, rB_src_BT, k0, d_cav_m, z_ex_opt_m)
    I_TM_v_BT = eml_profile_from_source_fields(
        z_eml_opt_m, z_ex_opt_m, k0,
        f_TM_v_BT.Ea_plus, f_TM_v_BT.Ea_minus, f_TM_v_BT.Eb_plus, f_TM_v_BT.Eb_minus,
    )
    I_iso_BT = (2.0 / 3.0) * (I_TE_h_BT + I_TM_h_BT) + (1.0 / 3.0) * I_TM_v_BT

    
    # --- (A2) phase3 BT-proxy profile across INTERNAL_ORDER (physical z axis)
    # This is the *unnormalized proxy* used inside the phase3 optimizer:
    #   E2_proxy(z) = |1+ru(z)|^2 |1+rd(z)|^2 / |1-ru(z)rd(z)|^2
    # where ru/rd are built from (rb_used, rt_used) and the optical coordinate zopt.
    th_nm = [float(d_best[k]) for k in INTERNAL_ORDER]
    bounds = np.cumsum([0.0] + th_nm)
    z_lcav_nm_BT = np.linspace(0.0, bounds[-1], 7001)

    nre = {k: float(np.real(n0[k])) for k in INTERNAL_ORDER}
    zopt_lcav_nm_BT = np.zeros_like(z_lcav_nm_BT)
    zc = 0.0
    sacc = 0.0
    idx_start = 0
    for name in INTERNAL_ORDER:
        dn = float(d_best[name])
        idx_end = np.searchsorted(z_lcav_nm_BT, zc + dn, side="right")
        zseg = z_lcav_nm_BT[idx_start:idx_end] - zc
        zopt_lcav_nm_BT[idx_start:idx_end] = sacc + nre[name] * zseg
        zc += dn
        sacc += nre[name] * dn
        idx_start = idx_end

    zopt_lcav_m_BT = zopt_lcav_nm_BT * 1e-9

    # Unnormalized phase3 proxy across LCAV (this is what the optimizer uses)
    ru_lcav = rb_used * np.exp(1j * 2.0 * k0 * zopt_lcav_m_BT)
    rd_lcav = rt_used * np.exp(1j * 2.0 * k0 * (d_cav_m - zopt_lcav_m_BT))
    D_lcav = (1.0 - (ru_lcav * rd_lcav))
    E2_proxy_lcav_BT = (np.abs(1.0 + ru_lcav) ** 2) * (np.abs(1.0 + rd_lcav) ** 2) / (np.abs(D_lcav) ** 2 + 1e-30)
    E2_proxy_lcav_BT_norm = E2_proxy_lcav_BT / (np.max(E2_proxy_lcav_BT) + 1e-30)
    # (Optional diagnostic) Evaluate a TE×h field-like profile using the BT cavity
    # source-plane fields. This is NOT the phase3 objective; the proxy above is.
    E_BT = np.zeros_like(zopt_lcav_m_BT, dtype=complex)
    for i, zz in enumerate(zopt_lcav_m_BT):
        dz = zz - z_ex_opt_m
        if zz <= z_ex_opt_m:
            E_BT[i] = f_TE_h_BT.Ea_plus * np.exp(1j * k0 * dz) + f_TE_h_BT.Ea_minus * np.exp(-1j * k0 * dz)
        else:
            E_BT[i] = f_TE_h_BT.Eb_plus * np.exp(1j * k0 * dz) + f_TE_h_BT.Eb_minus * np.exp(-1j * k0 * dz)
    E2_lcav_BT = np.abs(E_BT) ** 2
    E2_lcav_TE_h_BT_norm = E2_lcav_BT / (np.max(E2_lcav_BT) + 1e-30)
    E0_amp_norm_BT = float(np.sqrt(E2_lcav_TE_h_BT_norm[0]))

# --- (B) strict EML GPVM using physical EML boundaries (diagnostic)
    n_list, d_list_m, eml_index, z_ex_m = build_full_stack_from_phase3(n0, d_best)
    SA_TE, SB_TE = build_system_matrices_SA_SB(n_list, d_list_m, eml_index, z_ex_m, n_e, u_norm, lam_m, pol="TE")
    SA_TM, SB_TM = build_system_matrices_SA_SB(n_list, d_list_m, eml_index, z_ex_m, n_e, u_norm, lam_m, pol="TM")
    rA_TE, _tA_TE, rB_TE, _tB_TE = eq12_rt_from_SA_SB(SA_TE, SB_TE)
    rA_TM, _tA_TM, rB_TM, _tB_TM = eq12_rt_from_SA_SB(SA_TM, SB_TM)

    kz_e = kz(n0["EML"], n_e, u_norm, lam_m)
    d_eml_m = nm_to_m(float(d_best["EML"]))

    # z grid within EML (0 at left EML boundary)
    z_eml_m = np.linspace(0.0, d_eml_m, 2001)
    z_eml_nm = z_eml_m * 1e9

    # TE: horizontal dipole only
    f_TE_h = solve_source_plane_fields(A_TE_h.A_plus, A_TE_h.A_minus, rA_TE, rB_TE, kz_e, d_eml_m, z_ex_m)
    I_TE_h = eml_profile_from_source_fields(z_eml_m, z_ex_m, kz_e, f_TE_h.Ea_plus, f_TE_h.Ea_minus, f_TE_h.Eb_plus, f_TE_h.Eb_minus)

    # --- (B2) strict-EML LCAV profile across internal layers (same style as bottom-metal script)
    n_layers = {k: complex(n0[k]) for k in INTERNAL_ORDER}
    n_layers["EML"] = complex(np.real(n_layers["EML"]), 0.0)
    z_lcav_nm, E2_lcav_TE_h_norm, E0_amp_norm = lcav_profile_internal_from_fields(
        fields=f_TE_h,
        n_layers=n_layers,
        d_layers_nm=d_best,
        z_ex_m=z_ex_m,
        lambda0_m=lam_m,
        u=u_norm,
        pol="TE",
        layer_order=INTERNAL_ORDER,
        n_e=complex(n_e, 0.0),
        npts=7001,
    )

    # TM: horizontal and vertical
    f_TM_h = solve_source_plane_fields(A_TM_h.A_plus, A_TM_h.A_minus, rA_TM, rB_TM, kz_e, d_eml_m, z_ex_m)
    I_TM_h = eml_profile_from_source_fields(z_eml_m, z_ex_m, kz_e, f_TM_h.Ea_plus, f_TM_h.Ea_minus, f_TM_h.Eb_plus, f_TM_h.Eb_minus)
    f_TM_v = solve_source_plane_fields(A_TM_v.A_plus, A_TM_v.A_minus, rA_TM, rB_TM, kz_e, d_eml_m, z_ex_m)
    I_TM_v = eml_profile_from_source_fields(z_eml_m, z_ex_m, kz_e, f_TM_v.Ea_plus, f_TM_v.Ea_minus, f_TM_v.Eb_plus, f_TM_v.Eb_minus)

    # A simple isotropic blend (2/3 horizontal, 1/3 vertical). Horizontal contributes to both TE and TM.
    I_iso = (2.0 / 3.0) * (I_TE_h + I_TM_h) + (1.0 / 3.0) * I_TM_v

    # Normalize within EML for display
    def _norm(x):
        den = float(np.max(x)) + 1e-30
        return x / den

    # Normalize for plotting
    I_TE_h_BT_n = _norm(I_TE_h_BT)
    I_TM_h_BT_n = _norm(I_TM_h_BT)
    I_TM_v_BT_n = _norm(I_TM_v_BT)
    I_iso_BT_n = _norm(I_iso_BT)

    I_TE_h_n = _norm(I_TE_h)
    I_TM_h_n = _norm(I_TM_h)
    I_TM_v_n = _norm(I_TM_v)
    I_iso_n = _norm(I_iso)

    # Save outputs
    meta = {
        "lambda_nm": lambda_target_nm,
        "u_norm": u_norm,
        "n_e_re": n_e,
        "etl_best_nm": float(etl_best),
        "s_best": float(s_best),
        "d_best_nm": {k: float(v) for k, v in d_best.items()},
        "eml_thickness_nm": float(d_best["EML"]),
        "z_ex_phys_nm": float(z_ex_m * 1e9),
        "z_ex_opt_nm": float(z_ex_opt_m * 1e9),
        "LBT_opt_nm": float(np.real(LBT)),
        "rb_B_u0": [float(rb_used.real), float(rb_used.imag)],
        "rt_T_u0": [float(rt_used.real), float(rt_used.imag)],
        "phase3_score": float(score),
        "phase3_eml_center_nm": float(eml_center),
        "phase3_phi_e_650_rad": float(phi_e),  # legacy key name
        "phase3_phi_e_rad": float(phi_e),
        "phase3_phi_e_lambda_nm": float(lambda_target_nm),
    }
    best_json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # phase3-compatible (optical coordinate) profile samples over physical EML
    np.save(out_dir / "z_eml_phys_nm.npy", z_eml_phys_nm)
    np.save(out_dir / "I_iso_phase3BT.npy", I_iso_BT_n)
    np.save(out_dir / "I_TE_h_phase3BT.npy", I_TE_h_BT_n)
    np.save(out_dir / "I_TM_h_phase3BT.npy", I_TM_h_BT_n)
    np.save(out_dir / "I_TM_v_phase3BT.npy", I_TM_v_BT_n)

    # phase3-compatible LCAV profiles across INTERNAL_ORDER
    # (i) Unnormalized proxy used by the phase3 optimizer (recommended output)
    np.save(out_dir / 'z_lcav_nm_phase3BT.npy', z_lcav_nm_BT)
    np.save(out_dir / 'zopt_lcav_nm_phase3BT.npy', zopt_lcav_nm_BT)
    np.save(out_dir / 'E2_proxy_lcav_phase3BT_raw.npy', E2_proxy_lcav_BT)
    np.save(out_dir / 'E2_proxy_lcav_phase3BT_norm.npy', E2_proxy_lcav_BT_norm)
    # (ii) Optional diagnostic: a field-like TE×h profile constructed from source-plane fields
    np.save(out_dir / 'E2_lcav_TE_h_phase3BT_norm.npy', E2_lcav_TE_h_BT_norm)

    # strict-EML diagnostic profile (z in EML coordinates)
    np.save(out_dir / "z_eml_nm.npy", z_eml_nm)
    np.save(out_dir / "I_TE_h_strictEML.npy", I_TE_h_n)
    np.save(out_dir / "I_TM_h_strictEML.npy", I_TM_h_n)
    np.save(out_dir / "I_TM_v_strictEML.npy", I_TM_v_n)
    np.save(out_dir / "I_iso_strictEML.npy", I_iso_n)

    # strict-EML LCAV profile (TE×h) across INTERNAL_ORDER, normalized by max over LCAV
    np.save(out_dir / "z_lcav_nm.npy", z_lcav_nm)
    np.save(out_dir / "E2_lcav_TE_h_strictEML_norm.npy", E2_lcav_TE_h_norm)

    # Plot (A) phase3-compatible BT cavity
    plt.figure()
    plt.plot(z_eml_phys_nm, I_iso_BT_n, label="GPVM-cavity isotropic (norm)")
    plt.plot(z_eml_phys_nm, I_TE_h_BT_n, linestyle="--", label="TE (E) from horizontal")
    plt.plot(z_eml_phys_nm, I_TM_h_BT_n, linestyle=":", label="TM (H) from horizontal")
    plt.plot(z_eml_phys_nm, I_TM_v_BT_n, linestyle="-.", label="TM (H) from vertical")
    plt.axvline(float(eml_center), linestyle=":")
    plt.xlabel("physical z (nm) within EML")
    plt.ylabel("|field|^2 normalized")
    plt.title(f"GPVM phase3-compatible cavity @ {lambda_target_nm:.0f} nm, u={u_norm:.2f} (ETL={etl_best:.1f} nm, s={s_best:.4f})")
    plt.legend()
    plt.savefig(out_dir / "gpvm_eml_profiles_phase3BT.png", dpi=200, bbox_inches="tight")

    # Plot (B) strict EML
    plt.figure()
    plt.plot(z_eml_nm, I_iso_n, label="GPVM strict-EML isotropic (norm)")
    plt.plot(z_eml_nm, I_TE_h_n, linestyle="--", label="TE (E) from horizontal")
    plt.plot(z_eml_nm, I_TM_h_n, linestyle=":", label="TM (H) from horizontal")
    plt.plot(z_eml_nm, I_TM_v_n, linestyle="-.", label="TM (H) from vertical")
    plt.axvline(float(z_ex_m * 1e9), linestyle=":")
    plt.xlabel("z in EML (nm), 0=left boundary")
    plt.ylabel("|field|^2 normalized")
    plt.title(f"GPVM strict-EML @ {lambda_target_nm:.0f} nm, u={u_norm:.2f} (ETL={etl_best:.1f} nm, s={s_best:.4f})")
    plt.legend()
    plt.savefig(out_dir / "gpvm_eml_profiles_strictEML.png", dpi=200, bbox_inches="tight")

    # Plot (A2) phase3 BT-proxy LCAV profile (this matches the phase3 objective)
    th_nm = [float(d_best[k]) for k in INTERNAL_ORDER]
    bounds = np.cumsum([0.0] + th_nm)
    idx_eml = INTERNAL_ORDER.index('EML')
    z_eml0 = float(bounds[idx_eml])
    z_eml1 = float(bounds[idx_eml + 1])
    z_src_bt = float(eml_center)

    plt.figure(figsize=(10.5, 5.6))
    plt.plot(z_lcav_nm_BT, E2_proxy_lcav_BT_norm)
    # EML boundaries + center (requested)
    plt.axvline(z_eml0, linestyle='--')
    plt.axvline(z_eml1, linestyle='--')
    plt.axvline(z_src_bt, linestyle='--')
    plt.xlabel('z in Lcav (nm)')
    plt.ylabel('normalized proxy E2 (max over Lcav = 1)')
    plt.title(
        f'LCAV proxy profile (phase3-BT objective, u=0, λ={lambda_target_nm:.0f} nm)\n'
        f'Raw proxy saved as E2_proxy_lcav_phase3BT_raw.npy'
    )
    plt.tight_layout()
    plt.savefig(out_dir / 'gpvm_lcav_proxy_phase3BT.png', dpi=180)

    # Optional diagnostic plot: field-like TE×h LCAV profile (BT model)
    plt.figure(figsize=(10.5, 5.6))
    plt.plot(z_lcav_nm_BT, E2_lcav_TE_h_BT_norm)
    plt.axvline(z_eml0, linestyle='--')
    plt.axvline(z_eml1, linestyle='--')
    plt.axvline(z_src_bt, linestyle='--')
    plt.xlabel('z in Lcav (nm)')
    plt.ylabel('normalized |E|^2 (max over Lcav = 1)')
    plt.title(
        f'LCAV |E|^2 profile (phase3-BT field-like, TE×h, u=0, λ={lambda_target_nm:.0f} nm)\n'
        f'|E(z=0)| ≈ {E0_amp_norm_BT:.3e} (amplitude, normalized by max)'
    )
    plt.tight_layout()
    plt.savefig(out_dir / 'gpvm_lcav_field_phase3BT_TE_h.png', dpi=180)

    # Plot (C) strict-EML LCAV profile (TE×h), same style as bottom-metal script
    th_nm = [float(d_best[k]) for k in INTERNAL_ORDER]
    bounds = np.cumsum([0.0] + th_nm)
    idx_eml = INTERNAL_ORDER.index("EML")
    z_eml0 = float(bounds[idx_eml])
    z_eml1 = float(bounds[idx_eml + 1])
    z_src = z_eml0 + float(z_ex_m * 1e9)

    plt.figure(figsize=(10.5, 5.6))
    plt.plot(z_lcav_nm, E2_lcav_TE_h_norm)
    plt.axvline(z_eml0, linestyle="--")
    plt.axvline(z_eml1, linestyle="--")
    plt.axvline(z_src, linestyle="--")
    plt.xlabel("z in Lcav (nm)")
    plt.ylabel("normalized |E|^2 (max over Lcav = 1)")
    plt.title(
        f"LCAV |E|^2 profile (strict-EML, TE×h, u=0, λ={lambda_target_nm:.0f} nm)\n"
        f"|E(z=0)| ≈ {E0_amp_norm:.3e} (amplitude, normalized by max)"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "gpvm_lcav_profile_strictEML_TE_h.png", dpi=180)

    # Console summary
    print("[phase3-opt] best ETL_nm=%.3f, s=%.6f, score=%.6e" % (etl_best, s_best, score))
    print("[phase3-BT] rb_B(u=0)=%.6g%+.6gj, rt_T(u=0)=%.6g%+.6gj" % (rb_used.real, rb_used.imag, rt_used.real, rt_used.imag))
    print("[phase3-BT] rA_src(u=0)=%.6g%+.6gj, rB_src(u=0)=%.6g%+.6gj" % (rA_src_BT.real, rA_src_BT.imag, rB_src_BT.real, rB_src_BT.imag))
    print("[strict-EML] rA_TE=%.6g%+.6gj, rB_TE=%.6g%+.6gj" % (rA_TE.real, rA_TE.imag, rB_TE.real, rB_TE.imag))
    print("[strict-EML] rA_TM=%.6g%+.6gj, rB_TM=%.6g%+.6gj" % (rA_TM.real, rA_TM.imag, rB_TM.real, rB_TM.imag))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
