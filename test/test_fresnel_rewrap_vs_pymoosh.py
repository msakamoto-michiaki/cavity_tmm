# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from utils import require_pymoosh, check_close_complex, ensure_import_paths, is_verbose

def run(verbose: bool | None = None) -> list[str]:
    ensure_import_paths()
    import tmm_rewrap_utils_policyB as rw

    PyMoosh, Structure = require_pymoosh()

    lam = np.array([450.0, 550.0, 650.0], float)
    # representative refractive indices (including absorption)
    pairs = [
        (1.0+0j, 1.5+0j),
        (1.8+0.02j, 1.5+0j),
        (1.7+0j, 0.2+3.0j),
    ]
    u_list = [0.0, 0.5, 0.9]
    fails: list[str] = []
    if is_verbose(verbose):
        print("[TEST] Fresnel (rewrap convention) interface r vs PyMoosh")
        print("  compares: rw._interface_r  <->  PyMoosh.coefficient_S on a single interface")
        print("  grid    : lam_nm=%s, u=%s, pol in {TE(0),TM(1)}" % (lam.tolist(), u_list))
        print("  criterion: atol=1e-10 OR rtol=1e-10")
        print()
    for (n1,n2) in pairs:
        for u in u_list:
            for pol in [0,1]:
                r_int = rw._interface_r(n1, n2, u, lam, pol)
                r_pm = rw.pymoosh_stack_reflection(PyMoosh, Structure, n1, [], n2, lam, u=u, pol=pol)
                fails += check_close_complex(
                    name=f"rewrap interface r vs PyMoosh (pol={pol})",
                    x=r_int,
                    ref=r_pm,
                    atol=1e-10,
                    rtol=1e-10,
                    verbose=verbose,
                    context=f"n1={n1}, n2={n2}, u={u}, lam_nm={lam.tolist()}"
                )
    return fails
