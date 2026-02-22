

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict-EML GPVM: K(λ,u=0) with full polarization decomposition.

Computes (strict EML interface r_A/r_B):
  - K_TE_h(λ,u=0)  (Eq.26+Eq.30)
  - K_TM_h(λ,u=0)  (Eq.27+Eq.30)
  - K_TM_v(λ,u=0)  (Eq.28+Eq.30)  (should be ~0 at u=0)
and outputs:
  - K_iso      = (2/3)K_TE_h + (2/3)K_TM_h + (1/3)K_TM_v
  - K_TE_only  = (2/3)K_TE_h
  - K_TM_h_only= (2/3)K_TM_h
  - K_TM_v_only= (1/3)K_TM_v

Uses the phase3 best geometry at:
  out_gpvm_step2/best_geometry.json

Also replaces ITO index by PEC-like metal:
  n_ITO = 0.14 + 2000 i

Run:
  python gpvm_k_lambda_u0/run_gpvm_K_lambda_u0.py
"""

from gpvm_k_lambda_u0.run_gpvm_K_lambda_u0 import main

if __name__ == "__main__":
    main()
