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

from gpvm_K_lambda_kpar_map_strict.run_gpvm_K_lambda_kpar_map_strict import main

if __name__ == "__main__":
    main()
    
