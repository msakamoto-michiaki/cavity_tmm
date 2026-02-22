# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from utils import require_pymoosh, check_close_complex, ensure_import_paths, is_verbose, summarize_complex

def run(verbose: bool | None = None) -> list[str]:
    ensure_import_paths()
    import tmm_rewrap_utils_policyB as rw
    from oled_cavity_phase3p1_policyB import build_current_base

    PyMoosh, Structure = require_pymoosh()

    n0, d0 = build_current_base()
    lam = np.array([450.0, 550.0, 650.0], float)
    fails: list[str] = []
    if is_verbose(verbose):
        print("[TEST] BT terminal reflections vs PyMoosh (ref)")
        print("  compares: rw.terminal_reflections_BT (internal S-matrix)  <->  rw.terminal_reflections_BT_from_pymoosh")
        print("  outputs : rb_raw (complex), rt (complex), rb_used policy(phiB=pi) = -abs(rb_raw)")
        print("  grid    : lam_nm=%s, u in {0.0,0.7}, pol in {TE(0),TM(1)}" % (lam.tolist(),))
        print("  criterion: rb_raw/rt atol=1e-10 OR rtol=1e-10;  rb_used policy atol=1e-12")
        print()
    for u in [0.0, 0.7]:
        for pol in [0, 1]:
            rb_used_s, rt_s, rb_raw_s = rw.terminal_reflections_BT(n0=n0, d=d0, lam_nm=lam, u=u, pol=pol)
            rb_used_p, rt_p, rb_raw_p = rw.terminal_reflections_BT_from_pymoosh(PyMoosh, Structure, n0, d0, lam, u, pol)
            # raw reflections should match
            fails += check_close_complex(
                name=f"BT rb_raw vs PyMoosh (pol={pol})",
                x=rb_raw_s,
                ref=rb_raw_p,
                atol=1e-10,
                rtol=1e-10,
                verbose=verbose,
                context=f"u={u}, lam_nm={lam.tolist()}"
            )
            fails += check_close_complex(
                name=f"BT rt vs PyMoosh (pol={pol})",
                x=rt_s,
                ref=rt_p,
                atol=1e-10,
                rtol=1e-10,
                verbose=verbose,
                context=f"u={u}, lam_nm={lam.tolist()}"
            )
            # policy pi: rb_used should equal -abs(rb_raw)
            rb_should = -np.abs(rb_raw_s).astype(complex)
            if is_verbose(verbose):
                print(f"[INFO] rb_used policy check (u={u}, pol={pol}):")
                print(f"  rb_raw  sample: {summarize_complex(rb_raw_s)}")
                print(f"  rb_used sample: {summarize_complex(rb_used_s)}")
                print(f"  -abs(rb_raw)  : {summarize_complex(rb_should)}")
                print()
            fails += check_close_complex(
                name=f"BT rb_used == -abs(rb_raw) (pol={pol})",
                x=rb_used_s,
                ref=rb_should,
                atol=1e-12,
                rtol=1e-12,
                verbose=verbose,
                context=f"u={u}, lam_nm={lam.tolist()}"
            )
    return fails
