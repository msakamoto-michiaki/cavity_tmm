# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from utils import require_pymoosh, check_close_complex, ensure_import_paths, is_verbose

def run(verbose: bool | None = None) -> list[str]:
    ensure_import_paths()
    from gpvm.system_matrix import stack_transfer_matrix, rt_from_transfer_matrix
    PyMoosh, Structure = require_pymoosh()
    import tmm_rewrap_utils_policyB as rw

    # choose n_e=1 so gpvm u is identical to rewrap u (=k_parallel/k0)
    n_e = 1.0 + 0j
    u = 0.6
    lam_nm = 550.0
    lam_m = lam_nm * 1e-9

    n_inc = 1.0 + 0j
    n1 = 1.7 + 0.02j
    n2 = 1.3 + 0j
    n_sub = 1.5 + 0j
    d1_nm = 120.0
    d2_nm = 80.0

    # Build gpvm list (incident, layer1, layer2, substrate)
    n_list = [n_inc, n1, n2, n_sub]
    d_list_m = [d1_nm*1e-9, d2_nm*1e-9]

    fails: list[str] = []
    if is_verbose(verbose):
        print("[TEST] gpvm transfer-matrix stack r vs PyMoosh (ref)")
        print("  stack: incident -> (n1,d1) -> (n2,d2) -> substrate")
        print(f"  n_inc={n_inc}, n1={n1}, d1_nm={d1_nm}, n2={n2}, d2_nm={d2_nm}, n_sub={n_sub}")
        print(f"  lam_nm={lam_nm}, u={u}, n_e={n_e}")
        print("  note: TM has sign convention difference (GPVM TM = - PyMoosh/rewrap TM)")
        print("  criterion: atol=5e-9 OR rtol=5e-9")
        print()
    for pol_str, pol_int, expect_sign in [("TE",0, +1.0), ("TM",1, -1.0)]:
        M = stack_transfer_matrix(n_list=n_list, d_list_m=d_list_m, n_e=n_e, u=u, lambda0_m=lam_m, pol=pol_str)
        r_gpvm, _t_gpvm = rt_from_transfer_matrix(M)

        # PyMoosh reference for same physical stack using rewrap helper
        r_pm = rw.pymoosh_stack_reflection_scalar(
            PyMoosh, Structure,
            incident_n=n_inc,
            layers=[(n1,d1_nm),(n2,d2_nm)],
            substrate_n=n_sub,
            wavelength_nm=lam_nm,
            u=u,
            pol=pol_int,
        )
        r_ref = (expect_sign * r_pm)
        fails += check_close_complex(
            name=f"gpvm TMM r vs PyMoosh ({pol_str})",
            x=r_gpvm,
            ref=r_ref,
            atol=5e-9,
            rtol=5e-9,
            verbose=verbose,
            context=f"expect_sign={expect_sign:+.0f} applied to PyMoosh r"
        )
    return fails
