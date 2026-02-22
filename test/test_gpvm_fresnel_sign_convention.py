# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from utils import check_close_complex, ensure_import_paths, is_verbose

def run(verbose: bool | None = None) -> list[str]:
    ensure_import_paths()
    from gpvm.fresnel import fresnel_rt
    import tmm_rewrap_utils_policyB as rw

    n_e = 1.0 + 0j  # choose n_e=1 so gpvm u corresponds to rewrap u (=k_parallel/k0)
    lam_nm = np.array([450.0, 550.0, 650.0], float)
    lam_m = lam_nm * 1e-9
    u_list = [0.0, 0.4, 0.8]
    pairs = [(1.0+0j, 1.5+0j), (1.8+0.01j, 1.6+0j)]
    fails: list[str] = []
    if is_verbose(verbose):
        print("[TEST] GPVM Fresnel sign convention")
        print("  TE: gpvm.fresnel_rt should match rewrap interface r")
        print("  TM: gpvm.fresnel_rt should equal - rewrap interface r (convention difference)")
        print("  note: set n_e=1 so u is identical in both conventions")
        print("  criterion: atol=1e-10 OR rtol=1e-10")
        print()
    for (nj,nk) in pairs:
        for u in u_list:
            # TE: should match rewrap interface r
            r_gpvm_te = np.array([fresnel_rt(nj, nk, n_e, u, float(w), "TE").r for w in lam_m], complex)
            r_rw_te = rw._interface_r(nj, nk, u, lam_nm, 0)
            fails += check_close_complex(
                name="GPVM TE Fresnel == rewrap",
                x=r_gpvm_te,
                ref=r_rw_te,
                atol=1e-10,
                rtol=1e-10,
                verbose=verbose,
                context=f"nj={nj}, nk={nk}, u={u}, lam_nm={lam_nm.tolist()}"
            )
            # TM: gpvm has extra minus sign vs rewrap(Pymoosh conv)
            r_gpvm_tm = np.array([fresnel_rt(nj, nk, n_e, u, float(w), "TM").r for w in lam_m], complex)
            r_rw_tm = rw._interface_r(nj, nk, u, lam_nm, 1)
            fails += check_close_complex(
                name="GPVM TM Fresnel == -rewrap",
                x=r_gpvm_tm,
                ref=-r_rw_tm,
                atol=1e-10,
                rtol=1e-10,
                verbose=verbose,
                context=f"nj={nj}, nk={nk}, u={u}, lam_nm={lam_nm.tolist()}"
            )
    return fails
