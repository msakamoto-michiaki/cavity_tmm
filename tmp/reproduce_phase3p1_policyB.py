# -*- coding: utf-8 -*-
"""phase3p1 reproduce: S参照・rewrap維持で Proxy -> Green(Eq.54) へ置換

方針
----
- 参照面/ゲージ/反射係数生成/rewrap経路は phase2p8 と同一。
- 表示指標のみ変更:
  * F_alpha(lambda,u), alpha in {TE,TM}: Eq.(54)
      F_alpha = |1+rho_t|^2 |1+rho_b|^2 / |1-rho_t rho_b|^2
- K_h, K_v は Greenベース代替として
      K_h := F_TE,  K_v := F_TM
  を既定値にする（重み付き再合成は後段で実施可能）。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tmm_rewrap_utils_policyB import (
    effective_ru_rd_at_S_from_BT,
    terminal_reflections_BT,
    green_F_eq54_from_ru_rd,
)


def build_current_base():
    # constant complex n (same as original script)
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
    # "Current" thickness per original plot
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


_INTERNAL_ORDER = ["pHTL", "Rprime", "HTL", "EBL", "EML", "ETL"]


def optical_coord_u(n0, d, u):
    """Return (zSB, LBT) in the same convention as the original script.

    zSB: B->S (S=EML center) single-pass coordinate (units: nm * sqrt(n'^2-u^2))
    LBT: B->T single-pass coordinate (units: nm * sqrt(n'^2-u^2))
    """
    u = float(u)
    nre = {k: float(np.real(n0[k])) for k in _INTERNAL_ORDER}

    def kz_over_k0(nre_i):
        return np.sqrt((nre_i**2 - u**2) + 0j)  # complex allowed

    LBT = sum(float(d[k]) * kz_over_k0(nre[k]) for k in _INTERNAL_ORDER)

    zSB = (
        float(d["pHTL"]) * kz_over_k0(nre["pHTL"]) +
        float(d["Rprime"]) * kz_over_k0(nre["Rprime"]) +
        float(d["HTL"]) * kz_over_k0(nre["HTL"]) +
        float(d["EBL"]) * kz_over_k0(nre["EBL"]) +
        0.5 * float(d["EML"]) * kz_over_k0(nre["EML"])
    )
    return zSB, LBT


def ru_rd_from_BT(lam_nm, rb_B, rt_T, zSB, LBT):
    lam = np.asarray(lam_nm, float)
    beta = 4.0 * np.pi / lam
    zST = (LBT - zSB)
    ru_S = rb_B * np.exp(1j * beta * zSB)
    rd_S = rt_T * np.exp(1j * beta * zST)
    return ru_S, rd_S


def normalize_per_u(F):
    arr = np.asarray(F, float)
    den = np.max(arr, axis=1, keepdims=True) + 1e-30
    return arr / den


def reproduce_phase2_step2_u_dependence_sourceplane(
    out_dir="out_reproduce_step2_u_pymoosh_Splane_GreenEq54_TMMrewrap",
    cathode1_nm=30.0,
    lambda_min=450.0,
    lambda_max=800.0,
    lambda_samples=701,
    u_samples=481,
    kpar_max=3.0e7,
    kpar_samples=601,
    phi_b_pi_fixed=True,
    use_finetuned_thickness=True,
    do_consistency_check=True,
    # TM phase-policy switches (requested)
    tm_top_pi_shift=True,
    tm_bottom_pi_shift=False,
    tm_internal_pi_shift=True,
):
    # Reproducibility:
    # If out_dir is relative, write outputs next to this script (not to the caller's CWD).
    # If out_dir is absolute, respect it.
    out_dir_path = Path(out_dir)
    OUT = out_dir_path if out_dir_path.is_absolute() else (_BASE / out_dir_path)
    OUT.mkdir(parents=True, exist_ok=True)

    n0, d0 = build_current_base()

    # Phase policy used in this run:
    #   B boundary : TE/TM both follow phi_b_mode ('pi' by default)
    #   T boundary : TE raw, TM optionally pi-shift (tm_top_pi_shift)
    #   intermediate rewrap interfaces : TM optionally pi-shift per interface
    #     via tm_internal_pi_shift (r_int -> -r_int in wrap recursion).
    d = dict(d0)
    d["cathode1"] = float(cathode1_nm)

    if use_finetuned_thickness:
        # same as original reproduce script
        d["ETL"] = 48.2
        s = 0.3958
        d["HTL"] = d0["HTL"] * s
        d["Rprime"] = d0["Rprime"] * s

    lam = np.linspace(float(lambda_min), float(lambda_max), int(lambda_samples))
    n_ref = float(np.real(n0["EML"]))
    u_grid = np.linspace(0.0, 0.98 * n_ref, int(u_samples))

    # Raw Eq.(54)
    F_TE = np.zeros((len(u_grid), len(lam)), float)
    F_TM = np.zeros((len(u_grid), len(lam)), float)
    K_h = np.zeros((len(u_grid), len(lam)), float)
    K_v = np.zeros((len(u_grid), len(lam)), float)

    # also compute BT-version for validation if requested
    if do_consistency_check:
        F_TE_BT = np.zeros_like(F_TE)
        F_TM_BT = np.zeros_like(F_TM)

    check_rows = []

    for iu, u in enumerate(u_grid):
        zSB_u, LBT_u = optical_coord_u(n0, d, u)

        phi_mode = "pi" if phi_b_pi_fixed else "raw"
        rb_TE, rt_TE, _rb_raw_TE = terminal_reflections_BT(
            n0=n0,
            d=d,
            lam_nm=lam,
            u=float(u),
            pol=0,
            phi_b_mode=phi_mode,
            tm_top_pi_shift=tm_top_pi_shift,
            tm_bottom_pi_shift=tm_bottom_pi_shift,
        )
        rb_TM, rt_TM, _rb_raw_TM = terminal_reflections_BT(
            n0=n0,
            d=d,
            lam_nm=lam,
            u=float(u),
            pol=1,
            phi_b_mode=phi_mode,
            tm_top_pi_shift=tm_top_pi_shift,
            tm_bottom_pi_shift=tm_bottom_pi_shift,
        )

        # --- S-gauge Eq.(54) with TMM rewrap between S and B/T ---
        ruS_TE, rdS_TE = effective_ru_rd_at_S_from_BT(
            n0, d, lam, u, rb_TE, rt_TE, pol=0,
            tm_internal_pi_shift=tm_internal_pi_shift,
        )
        ruS_TM, rdS_TM = effective_ru_rd_at_S_from_BT(
            n0, d, lam, u, rb_TM, rt_TM, pol=1,
            tm_internal_pi_shift=tm_internal_pi_shift,
        )

        Fte = green_F_eq54_from_ru_rd(ruS_TE, rdS_TE, eta=0.0)
        Ftm = green_F_eq54_from_ru_rd(ruS_TM, rdS_TM, eta=0.0)

        F_TE[iu, :] = Fte
        F_TM[iu, :] = Ftm

        # phase3p1 policy
        K_h[iu, :] = Fte
        K_v[iu, :] = Ftm

        if do_consistency_check:
            # BT optical-coordinate expression should match S rewrap numerically
            ruBT_TE, rdBT_TE = ru_rd_from_BT(lam, rb_TE, rt_TE, zSB_u, LBT_u)
            ruBT_TM, rdBT_TM = ru_rd_from_BT(lam, rb_TM, rt_TM, zSB_u, LBT_u)

            Fte_BT = green_F_eq54_from_ru_rd(ruBT_TE, rdBT_TE, eta=0.0)
            Ftm_BT = green_F_eq54_from_ru_rd(ruBT_TM, rdBT_TM, eta=0.0)

            F_TE_BT[iu, :] = Fte_BT
            F_TM_BT[iu, :] = Ftm_BT

            # per-u max abs diff (raw & normalized)
            Fte_n = Fte / (np.max(Fte) + 1e-30)
            Fte_BTn = Fte_BT / (np.max(Fte_BT) + 1e-30)

            # TM note:
            # When tm_internal_pi_shift=True, S-rewrap(TM) intentionally applies
            # per-interface pi-shift in intermediate multilayers, while the BT direct
            # expression here does not. Therefore S-vs-BT(TM) is not a strict identity.
            if tm_internal_pi_shift:
                tm_raw_diff = np.nan
                tm_norm_diff = np.nan
            else:
                Ftm_n = Ftm / (np.max(Ftm) + 1e-30)
                Ftm_BTn = Ftm_BT / (np.max(Ftm_BT) + 1e-30)
                tm_raw_diff = float(np.max(np.abs(Ftm - Ftm_BT)))
                tm_norm_diff = float(np.max(np.abs(Ftm_n - Ftm_BTn)))

            check_rows.append({
                "u": float(u),
                "max_abs_diff_F_TE_raw": float(np.max(np.abs(Fte - Fte_BT))),
                "max_abs_diff_F_TM_raw": tm_raw_diff,
                "max_abs_diff_F_TE_norm": float(np.max(np.abs(Fte_n - Fte_BTn))),
                "max_abs_diff_F_TM_norm": tm_norm_diff,
            })

    # normalized maps for display
    F_TE_norm = normalize_per_u(F_TE)
    F_TM_norm = normalize_per_u(F_TM)
    K_h_norm = normalize_per_u(K_h)
    K_v_norm = normalize_per_u(K_v)

    # --- Consistency report ---
    if do_consistency_check:
        df_check = pd.DataFrame(check_rows)
        df_check.to_csv(OUT / "consistency_check_F_eq54.csv", index=False)

        # --- safe max helper (All-NaNでもwarningを出さない) ---
        def _safe_finite_max(series):
            arr = pd.to_numeric(series, errors="coerce").to_numpy(float)
            finite = np.isfinite(arr)
            return float(np.max(arr[finite])) if np.any(finite) else float("nan")

        max_te_raw = _safe_finite_max(df_check["max_abs_diff_F_TE_raw"])
        max_tm_raw = _safe_finite_max(df_check["max_abs_diff_F_TM_raw"])
        max_te     = _safe_finite_max(df_check["max_abs_diff_F_TE_norm"])
        max_tm     = _safe_finite_max(df_check["max_abs_diff_F_TM_norm"])

        #max_te_raw = float(df_check["max_abs_diff_F_TE_raw"].max())
        #max_tm_raw = float(np.nanmax(df_check["max_abs_diff_F_TM_raw"].to_numpy(float)))
        #max_te = float(df_check["max_abs_diff_F_TE_norm"].max())
        #max_tm = float(np.nanmax(df_check["max_abs_diff_F_TM_norm"].to_numpy(float)))

        with open(OUT / "consistency_check_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"max_abs_diff_F_TE_raw = {max_te_raw:.6e}\n")
            f.write(f"max_abs_diff_F_TM_raw = {max_tm_raw:.6e}\n")
            f.write(f"max_abs_diff_F_TE_norm = {max_te:.6e}\n")
            f.write(f"max_abs_diff_F_TM_norm = {max_tm:.6e}\n")
            if tm_internal_pi_shift:
                f.write("note_TM: TM S-vs-BT identity is not enforced because tm_internal_pi_shift=True\n")

        # difference heatmaps (normalized visual confirmation)
        plt.figure()
        plt.imshow(np.abs(normalize_per_u(F_TE) - normalize_per_u(F_TE_BT)), aspect='auto', origin='lower',
                   extent=[lam[0], lam[-1], u_grid[0], u_grid[-1]])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("u")
        plt.title("|F_TE_norm(S) - F_TE_norm(BT)| [Eq.54]")
        plt.colorbar()
        plt.savefig(OUT / "diff_heatmap_F_TE_norm.png", dpi=200, bbox_inches="tight")
        plt.close()

        if not tm_internal_pi_shift:
            plt.figure()
            plt.imshow(np.abs(normalize_per_u(F_TM) - normalize_per_u(F_TM_BT)), aspect='auto', origin='lower',
                       extent=[lam[0], lam[-1], u_grid[0], u_grid[-1]])
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("u")
            plt.title("|F_TM_norm(S) - F_TM_norm(BT)| [Eq.54]")
            plt.colorbar()
            plt.savefig(OUT / "diff_heatmap_F_TM_norm.png", dpi=200, bbox_inches="tight")
            plt.close()

    # --- Save arrays ---
    # phase3p1 primary outputs (raw Eq.54)
    np.save(OUT / "F_TE_eq54.npy", F_TE)
    np.save(OUT / "F_TM_eq54.npy", F_TM)
    np.save(OUT / "K_h_eq54.npy", K_h)
    np.save(OUT / "K_v_eq54.npy", K_v)

    # compatibility filenames
    np.save(OUT / "F_TE.npy", F_TE)
    np.save(OUT / "F_TM.npy", F_TM)
    np.save(OUT / "K_h.npy", K_h)
    np.save(OUT / "K_v.npy", K_v)

    # normalized arrays for plotting/debug
    np.save(OUT / "F_TE_norm.npy", F_TE_norm)
    np.save(OUT / "F_TM_norm.npy", F_TM_norm)
    np.save(OUT / "K_h_norm.npy", K_h_norm)
    np.save(OUT / "K_v_norm.npy", K_v_norm)

    pd.DataFrame({"lambda_nm": lam}).to_csv(OUT / "lambda_grid.csv", index=False)
    pd.DataFrame({"u": u_grid}).to_csv(OUT / "u_grid.csv", index=False)

    # heatmaps (normalized display)
    def save_heatmap(arr, fname, title):
        plt.figure()
        plt.imshow(arr, aspect='auto', origin='lower', extent=[lam[0], lam[-1], u_grid[0], u_grid[-1]])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("u")
        plt.title(title)
        plt.colorbar()
        plt.savefig(OUT / fname, dpi=200, bbox_inches="tight")
        plt.close()

    save_heatmap(F_TE_norm, "heatmap_F_TE.png", "F_TE norm (S-gauge, Eq.54)")
    save_heatmap(F_TM_norm, "heatmap_F_TM.png", "F_TM norm (S-gauge, Eq.54)")
    save_heatmap(K_h_norm, "heatmap_K_horizontal.png", "K_horizontal norm (phase3p1)")
    save_heatmap(K_v_norm, "heatmap_K_vertical.png", "K_vertical norm (phase3p1)")

    # --- Fig.4-style (lambda vs k_parallel) map for K_iso ---
    K_iso_u = (2.0 / 3.0) * K_h + (1.0 / 3.0) * K_v

    kpar = np.linspace(0.0, float(kpar_max), int(kpar_samples))
    Kiso_map = np.zeros((len(lam), len(kpar)), float)

    # interpolate along u for each wavelength: u_req = (lambda/2pi) * k_parallel
    for il, w in enumerate(lam):
        # NOTE: kpar is in 1/m while wavelength is in nm.
        # u = k_parallel / k0, k0 = 2π/λ  (λ must be in meters here)
        # => u_req = (λ[m]/2π) * k_parallel = (λ[nm]*1e-9/2π) * k_parallel
        u_req = (w * 1e-9 / (2.0 * np.pi)) * kpar
        Kiso_map[il, :] = np.interp(u_req, u_grid, K_iso_u[:, il], left=0.0, right=0.0)

    Kmax = float(np.max(Kiso_map))
    Kdisp = np.log10(np.maximum(Kiso_map / (Kmax + 1e-30), 1e-10))
    Kdisp = np.clip(Kdisp, -2.0, 0.0)

    # light lines
    def k0_of_lambda_nm(lam_nm):
        return 2.0 * np.pi / (lam_nm * 1e-9)

    n_air = 1.0
    n_sub = float(np.real(n0["substrate"]))
    n_org = float(np.real(n0["EML"]))
    k_air = np.array([n_air * k0_of_lambda_nm(x) for x in lam], float)
    k_sub = np.array([n_sub * k0_of_lambda_nm(x) for x in lam], float)
    k_org = np.array([n_org * k0_of_lambda_nm(x) for x in lam], float)

    plt.figure(figsize=(7.8, 5.8))
    extent_k = [kpar.min(), kpar.max(), lam.min(), lam.max()]
    plt.imshow(Kdisp, aspect="auto", origin="lower", extent=extent_k, vmin=-2, vmax=0)
    plt.xlabel("In-plane wavevector k$_\\parallel$ (m$^{-1}$)")
    plt.ylabel("Wavelength (nm)")
    plt.title("Fig.4-style map (phase3p1): log10(K_iso/K_max), phi_b=pi")
    cbar = plt.colorbar()
    cbar.set_label("log$_{10}$(K/K$_{max}$)")
    plt.plot(k_air, lam, "w-", linewidth=2.2, label="air light line")
    plt.plot(k_sub, lam, "w-", linewidth=2.2, label="substrate light line")
    plt.plot(k_org, lam, "w-", linewidth=2.2, label="organic light line (Re(n_EML)k0)")
    plt.legend(loc="upper right", framealpha=0.85, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_style_Kiso_map.png", dpi=240)
    plt.close()

    # save thickness and phase-policy switches
    pd.DataFrame([d]).to_csv(OUT / "thickness_used.csv", index=False)
    pd.DataFrame([{
        "phi_b_mode": ("pi" if phi_b_pi_fixed else "raw"),
        "tm_top_pi_shift": bool(tm_top_pi_shift),
        "tm_bottom_pi_shift": bool(tm_bottom_pi_shift),
        "tm_internal_pi_shift": bool(tm_internal_pi_shift),
    }]).to_csv(OUT / "phase_policy_used.csv", index=False)

    return str(OUT)


def main():
    reproduce_phase2_step2_u_dependence_sourceplane(
        # requested policy:
        # B: phi_B=pi for TE/TM, T: TE raw + TM pi-shift, internal: TM pi-shift
        do_consistency_check=False,
        tm_top_pi_shift=True,
        tm_bottom_pi_shift=False,
        tm_internal_pi_shift=True,
    )


if __name__ == "__main__":
    main()
