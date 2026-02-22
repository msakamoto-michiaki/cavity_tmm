#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate K(lambda,u=0) comparison:
(A) phase3-BT proxy vs (B) strict-EML GPVM.

- Uses GPVM Eq.(26) and Eq.(30) at u=0 (TE, horizontal dipole).
- Isotropic weighting: K_iso = (2/3) K_TE,h.
- Loads best geometry from --best-json (default: out_gpvm_step2/best_geometry.json).
- Replaces ITO index with: n = 0.14 + 2000i (PEC-like metal)

Run (from repo root):
  python gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py
  # or
  python gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py --best-json out_gpvm_step2/best_geometry_R.json --tag _R

Outputs are written into --outdir (default: gpvm_k_lambda_u0/).
"""

from gpvm_k_lambda_u0.run_gpvm_K_lambda_u0_A_vs_B import main

if __name__ == "__main__":
    main()
    
