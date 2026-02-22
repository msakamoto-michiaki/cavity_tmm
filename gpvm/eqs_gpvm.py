# gpvm/eqs_gpvm.py
# -*- coding: utf-8 -*-
"""GPVM core equations (Eq. 26–30) used across scripts.

This module centralizes the spectral power density terms (Pe) and the K-factor
used in the GPVM workflows.

Conventions:
- rA, rB are *source-plane referenced* complex reflection coefficients.
- kz_e is the EML longitudinal wavevector component (complex), with branch chosen
  so that evanescent waves decay away from the interface (Im(kz) >= 0).
- d_eml_m and z_ex_m are in meters; 0 < z_ex_m < d_eml_m.
"""

from __future__ import annotations

import numpy as np


def _D(rA: complex, rB: complex, kz_e: complex, d_eml_m: float) -> complex:
    """Common Fabry–Perot denominator D = 1 - rA rB exp(i 2 kz d)."""
    return 1.0 - (rA * rB * np.exp(1j * 2.0 * kz_e * d_eml_m))


def Pe_TE_h_eq26(rA: complex, rB: complex, kz_e: complex, d_eml_m: float, z_ex_m: float) -> float:
    """Eq. (26): spectral power density per unit solid angle for TE, horizontal dipole."""
    RA = abs(rA) ** 2
    RB = abs(rB) ** 2
    D = _D(rA, rB, kz_e, d_eml_m)

    # NOTE: Eq.(26) has (1 - R_B) multiplying the r_A term, and (1 - R_A) multiplying the r_B term.
    num_A = 1.0 + rA * np.exp(1j * 2.0 * kz_e * z_ex_m)
    num_B = 1.0 + rB * np.exp(1j * 2.0 * kz_e * (d_eml_m - z_ex_m))

    term_A = (1.0 - RB) * (abs(num_A / D) ** 2)
    term_B = (1.0 - RA) * (abs(num_B / D) ** 2)

    return float(np.real((3.0 / (16.0 * np.pi)) * (term_A + term_B)))


def Pe_TM_h_eq27(rA: complex, rB: complex, kz_e: complex, d_eml_m: float, z_ex_m: float, u: float) -> float:
    """Eq. (27): spectral power density per unit solid angle for TM, horizontal dipole."""
    RA = abs(rA) ** 2
    RB = abs(rB) ** 2
    D = _D(rA, rB, kz_e, d_eml_m)

    num_A = 1.0 + rA * np.exp(1j * 2.0 * kz_e * z_ex_m)
    num_B = 1.0 + rB * np.exp(1j * 2.0 * kz_e * (d_eml_m - z_ex_m))

    term_A = (1.0 - RB) * (abs(num_A / D) ** 2)
    term_B = (1.0 - RA) * (abs(num_B / D) ** 2)

    return float(np.real((3.0 / (16.0 * np.pi)) * (1.0 - u ** 2) * (term_A + term_B)))


def Pe_TM_v_eq28(rA: complex, rB: complex, kz_e: complex, d_eml_m: float, z_ex_m: float, u: float) -> float:
    """Eq. (28): spectral power density per unit solid angle for TM, vertical dipole."""
    RA = abs(rA) ** 2
    RB = abs(rB) ** 2
    D = _D(rA, rB, kz_e, d_eml_m)

    # NOTE: Eq.(28) uses MINUS signs in the numerators.
    num_A = 1.0 - rA * np.exp(1j * 2.0 * kz_e * z_ex_m)
    num_B = 1.0 - rB * np.exp(1j * 2.0 * kz_e * (d_eml_m - z_ex_m))

    term_A = (1.0 - RB) * (abs(num_A / D) ** 2)
    term_B = (1.0 - RA) * (abs(num_B / D) ** 2)

    return float(np.real((3.0 / (8.0 * np.pi)) * (u ** 2) * (term_A + term_B)))


def K_from_Pe_eq30(Pe: float, u: float) -> float:
    """Eq. (30): K(u) = π/(1-u^2) * Pe(u)."""
    denom = max(1e-30, (1.0 - u ** 2))
    return float(np.real(np.pi / denom * Pe))
