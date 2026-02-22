# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from utils import ensure_import_paths, max_abs, is_verbose, summarize_real, summarize_complex

def run(verbose: bool | None = None) -> list[str]:
    ensure_import_paths()
    import tmm_rewrap_utils_policyB as rw
    import oled_cavity_phase3p1_policyB as ph3

    n0, d0 = ph3.build_current_base()
    lam_target = 530.0
    pol = 0  # TE
    u = 0.0

    # Get terminal reflections at lambda_target (scalar) by internal S-matrix
    rb_used, rt, rb_raw = rw.terminal_reflections_BT(n0=n0, d=d0, lam_nm=np.array([lam_target], float), u=u, pol=pol)
    rb_used = complex(rb_used[0])
    rt_ = complex(rt[0])

    if is_verbose(verbose):
        print("[TEST] phase3 proxy E2 consistency (objective quantity)")
        print("  objective uses Eq66 proxy (E2_raw) and should be gauge-consistent with BT form")
        print(f"  lambda_target={lam_target} nm, pol={pol} (TE), u={u}")
        print(f"  terminal reflections (at target): rb_used={rb_used}, rt={rt_}, rb_raw={complex(rb_raw[0])}")
        print("  criteria:")
        print("    (a) max|E2_S_raw - E2_BT_raw| <= 1e-8")
        print("    (b) peak z within EML (allow tiny margin)")
        print("    (c) peak_err = |z_peak - eml_center| <= 5 nm (loose smoke)")
        print()

    # run optimize once (uses Eq66 proxy as objective)
    df, best = ph3.optimize_etl_then_s(n0, d0, cathode1_nm=30.0, lambda_target=lam_target, pol=pol)
    if best is None:
        return ["optimize_etl_then_s returned best=None"]
    (_score, etl, s_est, d_new, z_phys, zopt, E2z_S_abs, E2z_BT_abs,
     E2z_S_raw, E2z_BT_raw, eml_start, eml_end, eml_center, LBT, phi_e) = best

    # consistency of two E2_raw paths already computed in optimize
    diff = max_abs(np.asarray(E2z_S_raw) - np.asarray(E2z_BT_raw))
    fails: list[str] = []
    if diff > 1e-8:
        fails.append(f"proxy E2_raw mismatch (Eq66 vs BT gauge): max|Δ|={diff:.3e} (>1e-8)")

    if is_verbose(verbose):
        print(f"  max|Δ(E2_raw)| = {diff:.3e}")
        print(f"  E2_S_raw sample: {summarize_real(E2z_S_raw)}")
        print(f"  E2_BT_raw sample:{summarize_real(E2z_BT_raw)}")

    # peak should lie within EML (allow small numerical margin)
    z_peak = float(np.asarray(z_phys)[int(np.argmax(np.asarray(E2z_S_abs)))])
    if not (eml_start - 1e-3 <= z_peak <= eml_end + 1e-3):
        fails.append(f"peak not in EML: z_peak={z_peak:.3f} nm, EML=[{eml_start:.3f},{eml_end:.3f}] nm")

    # objective is peak_err = |z_peak - eml_center|
    peak_err = abs(z_peak - eml_center)
    if peak_err > 5.0:  # loose threshold for a smoke check
        fails.append(f"peak_err too large: {peak_err:.3f} nm (>5 nm)")

    if is_verbose(verbose):
        print(f"  z_peak={z_peak:.3f} nm")
        print(f"  EML: start={eml_start:.3f} nm, center={eml_center:.3f} nm, end={eml_end:.3f} nm")
        print(f"  peak_err={peak_err:.3f} nm")
        print()
    return fails
