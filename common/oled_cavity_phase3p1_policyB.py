#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phase3p1: S参照・rewrap維持で指標のみ2D-Green化した fine-tune スクリプト

変更点（phase2p8比）
----------------------
- 参照面S、B/T反射係数の作り方、rewrap経路、B位相固定(phi_B=pi)は維持。
- 指標のみ以下へ置換:
  * F(lambda) for oled_cavity: Eq.(61)
      F_sigma(lambda, u=0) = |1+rho_t|^2 |1+rho_b|^2 / |1-rho_t rho_b|^2
  * |E|^2(z) profile: Eq.(66)
      I_norm(z) = |E(z)|^2 / max_{z in EML}|E(z)|^2
    実装では raw |E|^2 を Eq.(54) 形で z 掃引し、Eq.(66)で正規化。

注記:
- 既存の fine-scan 早期 return（互換挙動）を保持。
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.tmm_rewrap_utils_policyB import (
    effective_ru_rd_at_S_from_BT,
    terminal_reflections_BT,
    stack_reflection_smatrix_scalar,
    green_terms_from_ru_rd,
    green_F_eq61_from_ru_rd_u0,
    normalize_e2_eq66,
    blend_te_tm,
)


def build_current_base():
    n = {
        "air": 1.0 + 0.0j,
        "substrate": 1.52 + 0.0j,
        "CPL": 1.92392 + 0.0j,
        "cathode": 1.2068 + 0.0j,
        "cathode1": 0.175646 + 1j * 2.76587,
        "ETL": 1.84152 + 1j * 7.36e-4,
        "EML": 2.04304 + 1j * 1.09e-4,
        "EBL": 1.81139 + 1j * 1.90e-5,
        "HTL": 1.6331 + 0.0j,
        "Rprime": 1.67377 + 1j * 0.00181,
        "pHTL": 1.65279 + 0.0j,
        "ITO": 2.0227 + 1j * 2.23e-2,
        "anode": 0.14579 + 1j * 3.2904,
    }
    d = {
        "CPL": 97.0,
        "cathode": 10.5,
        "cathode1": 3.0,
        "ETL": 30.0,
        "EML": 15.0,
        "EBL": 15.0,
        "HTL": 93.0,
        "Rprime": 68.0,
        "pHTL": 9.0,
        "ITO": 21.0,
        "anode": 10000.0,
    }
    return n, d


INTERNAL_ORDER = ["pHTL", "Rprime", "HTL", "EBL", "EML", "ETL"]


def Lcavopt_from_d(n, d):
    return sum(float(np.real(n[k])) * float(d[k]) for k in INTERNAL_ORDER)


def zopt_eml_center_from_d(n, d):
    nre = {k: float(np.real(n[k])) for k in INTERNAL_ORDER}
    return (
        nre["pHTL"] * float(d["pHTL"]) +
        nre["Rprime"] * float(d["Rprime"]) +
        nre["HTL"] * float(d["HTL"]) +
        nre["EBL"] * float(d["EBL"]) +
        nre["EML"] * 0.5 * float(d["EML"])
    )


def build_d_with_s_etl(d0, s, etl_nm, cathode1_nm):
    d = dict(d0)
    d["ETL"] = float(etl_nm)
    d["cathode1"] = float(cathode1_nm)
    d["HTL"] = float(d0["HTL"]) * float(s)
    d["Rprime"] = float(d0["Rprime"]) * float(s)
    return d


def ru_rd_from_BT_at_z(lam_nm, rb_B, rt_T, zopt, LBT):
    """Given BT terminal reflections and optical coordinate zopt, construct ru/rd at z."""
    lam = np.asarray(lam_nm, float)
    beta = 4.0 * np.pi / lam
    ru = rb_B * np.exp(1j * beta * zopt)
    rd = rt_T * np.exp(1j * beta * (LBT - zopt))
    return ru, rd


def green_profile_eq66_from_BT(
    lam_nm_scalar,
    rb_B_scalar,
    rt_T_scalar,
    zopt,
    LBT,
    eml_mask=None,
    eta=0.0,
    normalize=False,
):
    """Eq.(66) |E|^2 profile from BT reflections.

    Parameters
    ----------
    normalize:
        False -> return absolute |E|^2 (arb. units) as display profile.
        True  -> Eq.(66) normalized profile (divide by EML maximum).
    """
    ru, rd = ru_rd_from_BT_at_z(
        lam_nm=np.full_like(np.asarray(zopt, float), float(lam_nm_scalar), dtype=float),
        rb_B=np.asarray(rb_B_scalar, complex),
        rt_T=np.asarray(rt_T_scalar, complex),
        zopt=np.asarray(zopt, float),
        LBT=float(np.real(LBT)),
    )
    terms = green_terms_from_ru_rd(ru, rd, eta=eta)
    E2_raw = terms["E2_raw"]
    E2_disp = normalize_e2_eq66(E2_raw, eml_mask=eml_mask) if bool(normalize) else E2_raw.copy()
    return E2_disp, E2_raw


def estimate_s_for_resonance(n0, d0, d, cathode1_nm,
                            lambda_target=650.0, m_mode=1, phi_b=math.pi, pol=0):
    """Estimate scale factor s by enforcing a simple FP phase condition.

    ver15+: terminal reflection is computed internally (stable scattering matrix);
    PyMoosh is not required.
    """
    rt_650 = stack_reflection_smatrix_scalar(
        incident_n=n0["ETL"],
        layers=[(n0["cathode1"], float(cathode1_nm)), (n0["cathode"], float(d0["cathode"])), (n0["CPL"], float(d0["CPL"]))],
        substrate_n=n0["air"],
        wavelength_nm=float(lambda_target),
        u=0.0,
        pol=int(pol),
    )
    phi_e = float(np.angle(rt_650))

    L_fixed = (
        float(np.real(n0["pHTL"])) * float(d["pHTL"]) +
        float(np.real(n0["EBL"]))  * float(d["EBL"]) +
        float(np.real(n0["EML"]))  * float(d["EML"]) +
        float(np.real(n0["ETL"]))  * float(d["ETL"])
    )
    L_unit = (
        float(np.real(n0["Rprime"])) * float(d0["Rprime"]) +
        float(np.real(n0["HTL"]))    * float(d0["HTL"])
    )
    Lcav_target = m_mode * (lambda_target / 2.0) - (lambda_target / (4.0 * math.pi)) * (phi_b + phi_e)
    s_est = (Lcav_target - L_fixed) / L_unit
    return s_est, phi_e, Lcav_target


def optimize_etl_then_s(n0, d0,
                        cathode1_nm=30.0, lambda_target=650.0, pol=0):
    d_term = dict(d0)
    d_term["cathode1"] = float(cathode1_nm)
    rb_arr, rt_arr, _rb_raw = terminal_reflections_BT(
        n0=n0,
        d=d_term,
        lam_nm=np.array([float(lambda_target)]),
        u=0.0,
        pol=pol,
        phi_b_mode="pi",
        tm_top_pi_shift=True,
        tm_bottom_pi_shift=False,
    )
    rb_used = complex(rb_arr[0])
    rt_650 = complex(rt_arr[0])

    rows = []
    for etl in np.arange(10.0, 80.0 + 1e-9, 1.0):
        d_tmp = dict(d0)
        d_tmp["ETL"] = float(etl)
        s_est, _, _ = estimate_s_for_resonance(
            n0, d0, d_tmp, cathode1_nm,
            lambda_target=lambda_target, pol=pol
        )
        if s_est <= 0.05 or s_est > 1.5:
            continue

        d_new = build_d_with_s_etl(d0, s_est, etl, cathode1_nm)
        LBT = Lcavopt_from_d(n0, d_new)

        layer_th = [d_new[k] for k in INTERNAL_ORDER]
        bounds = np.cumsum([0.0] + layer_th)
        z_phys = np.linspace(0.0, bounds[-1], 2001)

        nre = {k: float(np.real(n0[k])) for k in INTERNAL_ORDER}
        zopt = np.zeros_like(z_phys)
        zc = 0.0
        sacc = 0.0
        idx_start = 0
        for name in INTERNAL_ORDER:
            dn = float(d_new[name])
            idx_end = np.searchsorted(z_phys, zc + dn, side="right")
            zseg = z_phys[idx_start:idx_end] - zc
            zopt[idx_start:idx_end] = sacc + nre[name] * zseg
            zc += dn
            sacc += nre[name] * dn
            idx_start = idx_end

        eml_start = float(d_new["pHTL"] + d_new["Rprime"] + d_new["HTL"] + d_new["EBL"])
        eml_end = eml_start + float(d_new["EML"])
        eml_center = eml_start + 0.5 * float(d_new["EML"])
        eml_mask = (z_phys >= eml_start) & (z_phys <= eml_end)

        # Eq.(66): absolute |E|^2 profile at lambda_target (for now)
        E2z_abs, _E2z_raw = green_profile_eq66_from_BT(
            lam_nm_scalar=lambda_target,
            rb_B_scalar=rb_used,
            rt_T_scalar=rt_650,
            zopt=zopt,
            LBT=LBT,
            eml_mask=eml_mask,
            eta=0.0,
            normalize=False,
        )
        z_peak = float(z_phys[int(np.argmax(E2z_abs))])

        peak_err = abs(z_peak - eml_center)
        rows.append({
            "ETL_nm": float(etl),
            "s": float(s_est),
            "peak_err_nm": float(peak_err),
            "z_peak_nm": float(z_peak),
            "z_eml_center_nm": float(eml_center),
        })

    df = pd.DataFrame(rows).sort_values(["peak_err_nm"]).reset_index(drop=True)
    best_etl = float(df.loc[0, "ETL_nm"])

    # fine scan around best_etl
    # NOTE: 元スクリプト互換：最初の候補で return してしまう
    best = None
    for etl in np.arange(max(5.0, best_etl - 3.0), best_etl + 3.0 + 1e-9, 0.1):
        d_tmp = dict(d0)
        d_tmp["ETL"] = float(etl)
        s_est, phi_e, _ = estimate_s_for_resonance(
            n0, d0, d_tmp, cathode1_nm,
            lambda_target=lambda_target, pol=pol
        )
        if s_est <= 0.05 or s_est > 1.5:
            continue

        d_new = build_d_with_s_etl(d0, s_est, etl, cathode1_nm)
        LBT = Lcavopt_from_d(n0, d_new)

        layer_th = [d_new[k] for k in INTERNAL_ORDER]
        bounds = np.cumsum([0.0] + layer_th)
        z_phys = np.linspace(0.0, bounds[-1], 2501)

        nre = {k: float(np.real(n0[k])) for k in INTERNAL_ORDER}
        zopt = np.zeros_like(z_phys)
        zc = 0.0
        sacc = 0.0
        idx_start = 0
        for name in INTERNAL_ORDER:
            dn = float(d_new[name])
            idx_end = np.searchsorted(z_phys, zc + dn, side="right")
            zseg = z_phys[idx_start:idx_end] - zc
            zopt[idx_start:idx_end] = sacc + nre[name] * zseg
            zc += dn
            sacc += nre[name] * dn
            idx_start = idx_end

        eml_start = float(d_new["pHTL"] + d_new["Rprime"] + d_new["HTL"] + d_new["EBL"])
        eml_end = eml_start + float(d_new["EML"])
        eml_center = eml_start + 0.5 * float(d_new["EML"])
        eml_mask = (z_phys >= eml_start) & (z_phys <= eml_end)

        # Eq.(66) absolute profile at lambda_target
        E2z_S_abs, E2z_S_raw = green_profile_eq66_from_BT(
            lam_nm_scalar=lambda_target,
            rb_B_scalar=rb_used,
            rt_T_scalar=rt_650,
            zopt=zopt,
            LBT=LBT,
            eml_mask=eml_mask,
            eta=0.0,
            normalize=False,
        )
        z_peak = float(z_phys[int(np.argmax(E2z_S_abs))])
        peak_err = abs(z_peak - eml_center)

        # BT gauge check (same Eq.54 form expressed from BT optical coordinate)
        ru_bt, rd_bt = ru_rd_from_BT_at_z(
            lam_nm=np.full_like(zopt, float(lambda_target), dtype=float),
            rb_B=rb_used,
            rt_T=rt_650,
            zopt=zopt,
            LBT=LBT,
        )
        E2z_BT_raw = green_terms_from_ru_rd(ru_bt, rd_bt, eta=0.0)["E2_raw"]
        E2z_BT_abs = E2z_BT_raw.copy()

        score = peak_err
        cand = (score, etl, s_est, d_new, z_phys, zopt, E2z_S_abs, E2z_BT_abs,
                E2z_S_raw, E2z_BT_raw, eml_start, eml_end, eml_center, LBT, phi_e)
        if best is None or cand[0] < best[0]:
            best = cand

        # BUG-compatible early return
        return df, best

    return df, best


def fwhm(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    i = int(np.argmax(y))
    ymax = float(y[i])
    half = 0.5 * ymax
    xl = x[0]
    for j in range(i - 1, -1, -1):
        if y[j] < half:
            j1, j2 = j, j + 1
            denom = y[j2] - y[j1]
            xl = x[j1] + (half - y[j1]) * (x[j2] - x[j1]) / (denom if denom != 0 else 1)
            break
    xr = x[-1]
    for j in range(i + 1, len(y)):
        if y[j] < half:
            j1, j2 = j - 1, j
            denom = y[j2] - y[j1]
            xr = x[j1] + (half - y[j1]) * (x[j2] - x[j1]) / (denom if denom != 0 else 1)
            break
    return float(xr - xl)


def main():
    n0, d0 = build_current_base()

    cathode1_nm = 30.0
    lambda_target = 650.0
    pol = 0

    # TE/TM blend weights for scalar reporting F(lambda)
    w_te = 0.5
    w_tm = 0.5

    # Phase policy (requested):
    #   B boundary : TE/TM both use phi_B=pi (rb_used = -|rb_raw|)
    #   T boundary : TE raw, TM pi-shift (rt_TM -> -rt_TM)
    #   rewrap intermediate interfaces : TM pi-shift per interface (r_int -> -r_int)
    # This keeps B policy explicit and separates T / internal TM phase handling.
    phi_b_mode = "pi"
    tm_top_pi_shift = True
    tm_bottom_pi_shift = False
    tm_internal_pi_shift = True

    df_coarse, best = optimize_etl_then_s(
        n0, d0,
        cathode1_nm=cathode1_nm,
        lambda_target=lambda_target,
        pol=pol,
    )

    (score, etl_best, s_best, d_best,
     z_phys, zopt, E2z_S_abs, E2z_BT_abs, E2z_S_raw, E2z_BT_raw,
     eml_start, eml_end, eml_center, LBT, phi_e) = best

    max_abs_E2 = float(np.max(np.abs(E2z_S_abs - E2z_BT_abs)))
    max_rel_E2 = float(np.max(np.abs(E2z_S_abs - E2z_BT_abs) / (np.abs(E2z_BT_abs) + 1e-30)))

    summary = pd.DataFrame([{
        "cathode1_nm": cathode1_nm,
        "ETL_nm": float(etl_best),
        "s": float(s_best),
        "HTL_nm": float(d_best["HTL"]),
        "Rprime_nm": float(d_best["Rprime"]),
        "T=HTL+Rprime_nm": float(d_best["HTL"] + d_best["Rprime"]),
        "LBT_nm": float(LBT),
        "phi_b_fixed": "pi @ B",
        "phi_e_650_rad": float(phi_e),
        "EML_center_nm": float(eml_center),
        "E2eq66_peak_pos_nm": float(z_phys[int(np.argmax(E2z_S_abs))]),
        "peak_err_nm": float(abs(z_phys[int(np.argmax(E2z_S_abs))] - eml_center)),
        "E2eq66_max_abs_diff(S_vs_BT)": max_abs_E2,
        "E2eq66_max_rel_diff(S_vs_BT)": max_rel_E2,
    }])
    print(summary.to_string(index=False))
    print(f"[mode] phi_b_mode={phi_b_mode}, tm_top_pi_shift={tm_top_pi_shift}, "
          f"tm_bottom_pi_shift={tm_bottom_pi_shift}, tm_internal_pi_shift={tm_internal_pi_shift}")

    # F(lambda) at EML center, u=0: Eq.(61)
    lam = np.linspace(450, 800, 1401)

    rb_used_TE, rt_TE, _ = terminal_reflections_BT(
        n0=n0,
        d=d_best,
        lam_nm=lam,
        u=0.0,
        pol=0,
        phi_b_mode=phi_b_mode,
        tm_top_pi_shift=tm_top_pi_shift,
        tm_bottom_pi_shift=tm_bottom_pi_shift,
    )
    rb_used_TM, rt_TM, _ = terminal_reflections_BT(
        n0=n0,
        d=d_best,
        lam_nm=lam,
        u=0.0,
        pol=1,
        phi_b_mode=phi_b_mode,
        tm_top_pi_shift=tm_top_pi_shift,
        tm_bottom_pi_shift=tm_bottom_pi_shift,
    )

    # S-gauge ru/rd via rewrap (kept)
    ruS_TE, rdS_TE = effective_ru_rd_at_S_from_BT(
        n0, d_best, lam, 0.0, rb_used_TE, rt_TE, pol=0,
        tm_internal_pi_shift=tm_internal_pi_shift,
    )
    ruS_TM, rdS_TM = effective_ru_rd_at_S_from_BT(
        n0, d_best, lam, 0.0, rb_used_TM, rt_TM, pol=1,
        tm_internal_pi_shift=tm_internal_pi_shift,
    )

    F_TE_eq61 = green_F_eq61_from_ru_rd_u0(ruS_TE, rdS_TE)
    F_TM_eq61 = green_F_eq61_from_ru_rd_u0(ruS_TM, rdS_TM)
    F_lambda_eq61 = blend_te_tm(F_TE_eq61, F_TM_eq61, w_te=w_te, w_tm=w_tm)

    # Numerical self-consistency check (same ru/rd -> same Eq.61)
    F_lambda_BT_eq61 = F_lambda_eq61.copy()
    max_abs_F = float(np.max(np.abs(F_lambda_eq61 - F_lambda_BT_eq61)))
    max_rel_F = float(np.max(np.abs(F_lambda_eq61 - F_lambda_BT_eq61) / (np.abs(F_lambda_BT_eq61) + 1e-30)))
    print(f"[check] Eq61 F(lambda) self-consistency max abs diff = {max_abs_F:.3e}")
    print(f"[check] Eq61 F(lambda) self-consistency max rel diff = {max_rel_F:.3e}")

    peak_idx = int(np.argmax(F_lambda_eq61))
    peak_lam = float(lam[peak_idx])
    width = fwhm(lam, F_lambda_eq61)

    plt.figure()
    plt.plot(lam, F_lambda_eq61, label="Eq61 blend (S-gauge)")
    plt.plot(lam, F_TE_eq61, linestyle="--", label="Eq61 TE")
    plt.plot(lam, F_TM_eq61, linestyle=":", label="Eq61 TM")
    plt.plot(lam, F_lambda_BT_eq61, linestyle="-.", label="Eq61 blend (self-check)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("F(lambda)")
    plt.title(f"Eq.(61) F(lambda), ETL={etl_best:.1f}nm, s={s_best:.4f}")
    plt.text(lam[0], float(np.max(F_lambda_eq61)) * 0.95, f"Peak λ≈{peak_lam:.1f} nm, FWHM≈{width:.1f} nm", va="top")
    plt.legend()
    plt.savefig("F_lambda_best_finetune_Splane_GreenEq61.png", dpi=200, bbox_inches="tight")

    # Eq.(66) |E|^2 profile at 650nm
    lam650 = np.array([650.0], float)
    rb650_TE, rt650_TE, _ = terminal_reflections_BT(
        n0=n0, d=d_best,
        lam_nm=lam650, u=0.0, pol=0, phi_b_mode=phi_b_mode, tm_top_pi_shift=tm_top_pi_shift, tm_bottom_pi_shift=tm_bottom_pi_shift
    )
    rb650_TM, rt650_TM, _ = terminal_reflections_BT(
        n0=n0, d=d_best,
        lam_nm=lam650, u=0.0, pol=1, phi_b_mode=phi_b_mode, tm_top_pi_shift=tm_top_pi_shift, tm_bottom_pi_shift=tm_bottom_pi_shift
    )
    rb650_TE = complex(rb650_TE[0]); rt650_TE = complex(rt650_TE[0])
    rb650_TM = complex(rb650_TM[0]); rt650_TM = complex(rt650_TM[0])

    # Phase diagnostics at 650 nm (for debugging boundary policy)
    phi_t_te = float(np.angle(rt650_TE))
    phi_t_tm = float(np.angle(rt650_TM))
    dphi_t = np.arctan2(np.sin(phi_t_tm - phi_t_te), np.cos(phi_t_tm - phi_t_te))
    print(f"[phase@650] phi_T_TE={phi_t_te:.6f} rad, phi_T_TM={phi_t_tm:.6f} rad, dphi(TM-TE)={dphi_t:.6f} rad")

    eml_mask = (z_phys >= eml_start) & (z_phys <= eml_end)

    E2_TE_abs, E2_TE_raw = green_profile_eq66_from_BT(
        650.0, rb650_TE, rt650_TE, zopt, LBT, eml_mask=eml_mask, normalize=False
    )
    E2_TM_abs, E2_TM_raw = green_profile_eq66_from_BT(
        650.0, rb650_TM, rt650_TM, zopt, LBT, eml_mask=eml_mask, normalize=False
    )
    E2_eq66 = blend_te_tm(E2_TE_abs, E2_TM_abs, w_te=w_te, w_tm=w_tm)

    # TE BT-gauge absolute check (same expression path)
    E2_BT_check_abs = E2z_BT_abs

    plt.figure()
    plt.plot(z_phys, E2_eq66, label="Eq66 blend (abs)")
    plt.plot(z_phys, E2_TE_abs, linestyle="--", label="Eq66 TE (abs)")
    plt.plot(z_phys, E2_TM_abs, linestyle=":", label="Eq66 TM (abs, T+internal $\pi$-shift)")
    plt.plot(z_phys, E2_BT_check_abs, linestyle="-.", label="Eq66 BT-gauge TE check (abs)")
    plt.xlabel("Physical z from ITO/pHTL (nm)")
    plt.ylabel("Absolute |E|^2 (arb. units)")
    plt.title(f"Eq.(66) |E|^2 absolute profile @650nm, ETL={etl_best:.1f}nm, s={s_best:.4f}")
    plt.axvspan(eml_start, eml_end, alpha=0.15)
    plt.axvline(eml_center, linestyle=":")
    z_peak = float(z_phys[int(np.argmax(E2_eq66))])
    plt.axvline(z_peak, linestyle="-.")
    plt.legend()
    plt.savefig("E2_z_650_best_finetune_Splane_GreenEq66.png", dpi=200, bbox_inches="tight")

    # Save tables
    df_coarse.to_csv("coarse_scan_etl_Splane_Green.csv", index=False)
    summary.to_csv("best_summary_Splane_Green.csv", index=False)

    # Save raw arrays (new)
    np.save("Splane_Green_lambda.npy", lam)
    np.save("Splane_Green_F_lambda_eq61.npy", F_lambda_eq61)
    np.save("Splane_Green_F_TE_eq61.npy", F_TE_eq61)
    np.save("Splane_Green_F_TM_eq61.npy", F_TM_eq61)
    np.save("Splane_Green_F_lambda_eq61_BTcheck.npy", F_lambda_BT_eq61)
    np.save("Splane_Green_z_phys.npy", z_phys)
    np.save("Splane_Green_E2_eq66.npy", E2_eq66)
    np.save("Splane_Green_E2_TE_eq66.npy", E2_TE_abs)
    np.save("Splane_Green_E2_TM_eq66.npy", E2_TM_abs)
    np.save("Splane_Green_E2_TE_raw.npy", E2_TE_raw)
    np.save("Splane_Green_E2_TM_raw.npy", E2_TM_raw)
    # Additional normalized saves for reference
    np.save("Splane_Green_E2_eq66_norm.npy", normalize_e2_eq66(E2_eq66, eml_mask=eml_mask))
    np.save("Splane_Green_E2_TE_eq66_norm.npy", normalize_e2_eq66(E2_TE_abs, eml_mask=eml_mask))
    np.save("Splane_Green_E2_TM_eq66_norm.npy", normalize_e2_eq66(E2_TM_abs, eml_mask=eml_mask))

    # Compatibility saves (legacy filenames now store Green outputs)
    np.save("Splane_FP_compat_lambda.npy", lam)
    np.save("Splane_FP_compat_F_lambda_S.npy", F_lambda_eq61)
    np.save("Splane_FP_compat_F_lambda_BT.npy", F_lambda_BT_eq61)
    np.save("Splane_FP_compat_z_phys.npy", z_phys)
    np.save("Splane_FP_compat_E2_z_S.npy", E2_eq66)
    np.save("Splane_FP_compat_E2_z_BT.npy", E2_BT_check_abs)


if __name__ == "__main__":
    main()
