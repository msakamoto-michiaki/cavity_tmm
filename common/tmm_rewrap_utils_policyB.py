# -*- coding: utf-8 -*-
"""tmm_rewrap_utils.py

Rewrap utilities for the OLED cavity proxy (FP) model.

This module now centralizes three processing steps that were previously duplicated
across scripts:

1) PyMoosh terminal-reflection evaluation at B/T boundaries
2) u -> theta conversion with a fixed complex branch
3) Bottom boundary phase policy (phi_B), including the project default phi_B = pi

Boundary definitions (intentionally preserved)
---------------------------------------------
B-side terminal reflection r_B is defined at pHTL|ITO seen from pHTL.
T-side terminal reflection r_T is defined at ETL|cathode1 seen from ETL.

Rewrap paths (intentionally preserved)
--------------------------------------
(i) Bottom: EML -> EBL -> HTL -> Rprime -> pHTL -> ITO (terminal)
(ii) Top   : EML -> ETL -> cathode1 (terminal)

Important conventions (match PyMoosh stable)
-------------------------------------------
In this project we represent materials by complex refractive index n(λ), but PyMoosh
`Structure(mats=...)` interprets a scalar material value as permittivity epsilon.
Therefore, whenever we compare to or emulate PyMoosh behavior, we use

  eps = n**2  (mu = 1)
  kz = k0 * sqrt(eps - u**2)

with the outgoing/decaying branch fixed by Im(kz) >= 0.

For interface reflection (seen from medium 1):
  r = (b1 - b2)/(b1 + b2)
  b = kz            (TE)
  b = kz/eps        (TM)

This TM convention is the one that makes rewrap match PyMoosh.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.scimath import sqrt as csqrt, arccos as carccos


# -----------------------------------------------------------------------------
# Basic optical helpers for rewrap core
# -----------------------------------------------------------------------------

def k0_of_lambda_nm(lam_nm: np.ndarray) -> np.ndarray:
    """Vacuum wavenumber in 1/nm."""
    lam = np.asarray(lam_nm, float)
    return 2.0 * np.pi / lam


def _fix_kz_branch(kz: np.ndarray) -> np.ndarray:
    """Enforce the outgoing/decaying branch: Im(kz) >= 0 elementwise."""
    kz = np.asarray(kz, complex)
    return kz * (1 - 2 * (np.imag(kz) < 0))


def kz_nm(n: complex, u: float, lam_nm: np.ndarray) -> np.ndarray:
    """kz in 1/nm for refractive index n, using eps=n^2 convention (mu=1)."""
    eps = (n + 0j) ** 2
    k0 = k0_of_lambda_nm(lam_nm)
    kz = k0 * np.sqrt(eps - (u + 0j) ** 2)
    return _fix_kz_branch(kz)


def _interface_r(n1: complex, n2: complex, u: float, lam_nm: np.ndarray, pol: int) -> np.ndarray:
    """Interface reflection at 1|2, seen from medium 1. pol: 0=TE, 1=TM."""
    eps1 = (n1 + 0j) ** 2
    eps2 = (n2 + 0j) ** 2
    kz1 = kz_nm(n1, u, lam_nm)
    kz2 = kz_nm(n2, u, lam_nm)

    if pol == 0:  # TE
        b1 = kz1
        b2 = kz2
    elif pol == 1:  # TM (PyMoosh convention)
        b1 = kz1 / (eps1 + 0j)
        b2 = kz2 / (eps2 + 0j)
    else:
        raise ValueError("pol must be 0 (TE) or 1 (TM)")

    return (b1 - b2) / (b1 + b2 + 0j)


# -----------------------------------------------------------------------------
# Stable multilayer reflection without PyMoosh (scattering-matrix cascade)
# -----------------------------------------------------------------------------

def _b_admittance(n: complex, u: float, lam_nm: np.ndarray, pol: int) -> np.ndarray:
    """Return b = kz (TE) or kz/eps (TM) using the same convention as _interface_r."""
    eps = (n + 0j) ** 2
    kzv = kz_nm(n, u, lam_nm)
    if int(pol) == 0:
        return kzv
    if int(pol) == 1:
        return kzv / (eps + 0j)
    raise ValueError("pol must be 0 (TE) or 1 (TM)")


def _S_interface(b1: complex, b2: complex) -> np.ndarray:
    """2x2 interface scattering matrix (left medium=1, right medium=2)."""
    den = (b1 + b2) + 0j
    return np.array(
        [
            [(b1 - b2) / den, (2.0 * b2) / den],
            [(2.0 * b1) / den, (b2 - b1) / den],
        ],
        dtype=complex,
    )


def _S_layer(t: complex) -> np.ndarray:
    """2x2 layer scattering matrix for a uniform layer of thickness d (encoded in t)."""
    return np.array([[0.0 + 0.0j, t], [t, 0.0 + 0.0j]], dtype=complex)


def _cascade_S(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cascade two 2x2 scattering matrices A then B (series connection)."""
    A00, A01, A10, A11 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    B00, B01, B10, B11 = B[0, 0], B[0, 1], B[1, 0], B[1, 1]
    t = 1.0 / ((1.0 - B00 * A11) + 0j)
    S00 = A00 + A01 * B00 * A10 * t
    S01 = A01 * B01 * t
    S10 = B10 * A10 * t
    S11 = B11 + A11 * B01 * B10 * t
    return np.array([[S00, S01], [S10, S11]], dtype=complex)


def stack_reflection_smatrix_spectrum(
    *,
    incident_n: complex,
    layers: list[tuple[complex, float]],
    substrate_n: complex,
    lam_nm: np.ndarray,
    u: float,
    pol: int,
) -> np.ndarray:
    """Compute multilayer reflection r(λ) using a stable scattering-matrix cascade.

    Parameters
    ----------
    incident_n:
        Refractive index of the semi-infinite incident medium.
    layers:
        List of (n_layer, d_layer_nm) from incident side to substrate side.
    substrate_n:
        Refractive index of the semi-infinite substrate medium.
    lam_nm:
        Wavelength grid (nm).
    u:
        In-plane variable **u = k_parallel / k0** (the same u used in this module).
    pol:
        0=TE, 1=TM (PyMoosh-compatible convention: b=kz for TE, b=kz/eps for TM).

    Returns
    -------
    r:
        Complex reflection coefficient seen from the incident medium.

    Notes
    -----
    This implementation mirrors the stable `coefficient_S` approach:
    layer propagation is represented by a scattering matrix with
      t = exp(i*kz*d)
    which decays for evanescent waves (Im(kz) >= 0), avoiding the exp(+Im*kz*d)
    blow-up inherent to raw transfer matrices.
    """
    lam = np.asarray(lam_nm, float)
    out = np.empty_like(lam, dtype=complex)
    # Material list including semi-infinite media
    n_all = [complex(incident_n)] + [complex(n) for (n, _d) in layers] + [complex(substrate_n)]
    d_all = [float(d) for (_n, d) in layers]

    for i, w in enumerate(lam):
        w_arr = np.array([float(w)], dtype=float)
        # Build total scattering matrix
        b0 = complex(_b_admittance(n_all[0], u, w_arr, pol=pol)[0])
        b1 = complex(_b_admittance(n_all[1], u, w_arr, pol=pol)[0])
        S = _S_interface(b0, b1)
        for j in range(1, len(n_all) - 1):
            # layer j thickness applies to medium n_all[j]
            if j - 1 < len(d_all):
                kzj = complex(kz_nm(n_all[j], u, w_arr)[0])
                t = complex(np.exp(1j * kzj * float(d_all[j - 1])))
                S = _cascade_S(S, _S_layer(t))
            # interface to next medium
            b_j = complex(_b_admittance(n_all[j], u, w_arr, pol=pol)[0])
            b_k = complex(_b_admittance(n_all[j + 1], u, w_arr, pol=pol)[0])
            S = _cascade_S(S, _S_interface(b_j, b_k))
        out[i] = S[0, 0]
    return out


def stack_reflection_smatrix_scalar(
    *,
    incident_n: complex,
    layers: list[tuple[complex, float]],
    substrate_n: complex,
    wavelength_nm: float,
    u: float,
    pol: int,
) -> complex:
    """Scalar wrapper for stack_reflection_smatrix_spectrum."""
    r = stack_reflection_smatrix_spectrum(
        incident_n=incident_n,
        layers=layers,
        substrate_n=substrate_n,
        lam_nm=np.array([float(wavelength_nm)], dtype=float),
        u=float(u),
        pol=int(pol),
    )
    return complex(r[0])


def terminal_reflections_BT(
    *,
    n0: dict,
    d: dict,
    lam_nm: np.ndarray,
    u: float,
    pol: int,
    phi_b_mode: str | float = "pi",
    tm_top_pi_shift: bool = False,
    tm_bottom_pi_shift: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (rb_used, rt, rb_raw) for the standard B/T terminal stacks **without PyMoosh**.

    This is the production path in ver15+: boundary reflections are evaluated by the
    internal stable scattering-matrix implementation. PyMoosh remains available only
    for optional test comparisons.
    """
    lam = np.asarray(lam_nm, float)
    # Bottom: r_B at pHTL|ITO seen from pHTL, through (ITO, anode) into substrate
    rb_raw = stack_reflection_smatrix_spectrum(
        incident_n=n0["pHTL"],
        layers=[(n0["ITO"], float(d["ITO"])), (n0["anode"], float(d["anode"]))],
        substrate_n=n0["substrate"],
        lam_nm=lam,
        u=float(u),
        pol=int(pol),
    )
    # Top: r_T at ETL|cathode1 seen from ETL, through (cathode1, cathode, CPL) into air
    rt = stack_reflection_smatrix_spectrum(
        incident_n=n0["ETL"],
        layers=[
            (n0["cathode1"], float(d["cathode1"])),
            (n0["cathode"], float(d["cathode"])),
            (n0["CPL"], float(d["CPL"])),
        ],
        substrate_n=n0["air"],
        lam_nm=lam,
        u=float(u),
        pol=int(pol),
    )

    if int(pol) == 1:
        if bool(tm_bottom_pi_shift):
            rb_raw = -rb_raw
        if bool(tm_top_pi_shift):
            rt = -rt

    rb_used = apply_bottom_phase_policy(rb_raw, phi_b_mode=phi_b_mode)
    return rb_used, rt, rb_raw


# -----------------------------------------------------------------------------
# Rewrap core
# -----------------------------------------------------------------------------

def wrap_reflection_from_terminal(
    n_inc: complex,
    layers: list[tuple[complex, float]],
    r_term: np.ndarray,
    u: float,
    lam_nm: np.ndarray,
    pol: int,
    tm_internal_pi_shift: bool = False,
) -> np.ndarray:
    """Rewrap a known termination reflection r_term through the provided layers.

    Parameters
    ----------
    n_inc:
        Refractive index of the incident medium at the top interface.
    layers:
        List of (n_layer, d_layer_nm) from top to bottom (closest to n_inc first).
    r_term:
        Reflection at (last layer)|(termination) seen from last layer (array over λ).
    u:
        Dimensionless in-plane variable u = k_parallel/k0.
    lam_nm:
        Wavelength grid in nm.
    pol:
        0=TE(s), 1=TM(p).
    tm_internal_pi_shift:
        If True, apply an additional π-phase shift to TM at each
        *intermediate rewrap interface* by using r_int -> -r_int when pol==1.

        This is an implementation policy requested for phase3 debugging:
          - B boundary: fixed separately by phi_b_mode='pi'
          - T boundary: handled in terminal_reflections_BT_*
          - intermediate multilayers: handled here per interface
    """
    lam = np.asarray(lam_nm, float)
    r_eff = np.asarray(r_term, complex)

    n_list = [n_inc] + [n for (n, _d) in layers]
    d_list = [0.0] + [float(d) for (_n, d) in layers]

    N = len(layers)
    for j in range(N - 1, -1, -1):
        n1 = n_list[j]
        n2 = n_list[j + 1]
        d2 = d_list[j + 1]
        phase = np.exp(2j * kz_nm(n2, u, lam) * d2)
        r_int = _interface_r(n1, n2, u, lam, pol)

        # Optional TM-only per-interface pi-shift for intermediate multilayers
        # requested by user: r_int(TM) -> -r_int(TM).
        # TE path is unchanged.
        if bool(tm_internal_pi_shift) and int(pol) == 1:
            r_int = -r_int

        r_eff = (r_int + r_eff * phase) / (1.0 + r_int * r_eff * phase + 0j)

    return r_eff


def effective_ru_rd_at_S_from_BT(
    n0: dict,
    d: dict,
    lam_nm: np.ndarray,
    u: float,
    rB_pol: np.ndarray,
    rT_pol: np.ndarray,
    pol: int,
    S_at_eml_center: bool = True,
    tm_internal_pi_shift: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute effective (ru_S, rd_S) at source plane S using rewrap.

    Boundary definition (kept unchanged):
      B = ITO/pHTL     rB_pol defined at pHTL|ITO, seen from pHTL
      T = ETL/cathode1 rT_pol defined at ETL|cathode1, seen from ETL

    Rewrap paths (kept unchanged):
      Bottom: EML -> EBL -> HTL -> Rprime -> pHTL -> ITO
      Top   : EML -> ETL -> cathode1

    We compute the reflections seen at the EML boundaries and then propagate to S.

    Parameters
    ----------
    tm_internal_pi_shift:
      If True and pol==TM, apply per-interface TM pi-shift (r_int->-r_int)
      in the intermediate multilayers during rewrap.
    """
    lam = np.asarray(lam_nm, float)

    # Bottom side (EML -> ... -> pHTL | ITO)
    layers_bottom = [
        (n0["EBL"], d["EBL"]),
        (n0["HTL"], d["HTL"]),
        (n0["Rprime"], d["Rprime"]),
        (n0["pHTL"], d["pHTL"]),
    ]
    r_eml_ebl = wrap_reflection_from_terminal(
        n_inc=n0["EML"],
        layers=layers_bottom,
        r_term=rB_pol,
        u=u,
        lam_nm=lam,
        pol=pol,
        tm_internal_pi_shift=tm_internal_pi_shift,
    )

    # Top side (EML -> ETL | cathode1)
    layers_top = [
        (n0["ETL"], d["ETL"]),
    ]
    r_eml_etl = wrap_reflection_from_terminal(
        n_inc=n0["EML"],
        layers=layers_top,
        r_term=rT_pol,
        u=u,
        lam_nm=lam,
        pol=pol,
        tm_internal_pi_shift=tm_internal_pi_shift,
    )

    # Shift from EML boundary to S inside EML
    if not S_at_eml_center:
        raise NotImplementedError("Only S at EML center is implemented.")
    dz = 0.5 * float(d["EML"])  # nm
    phase_S = np.exp(2j * kz_nm(n0["EML"], u, lam) * dz)

    ru_S = r_eml_ebl * phase_S
    rd_S = r_eml_etl * phase_S
    return ru_S, rd_S


def fp_filter_from_ru_rd(ru_S: np.ndarray, rd_S: np.ndarray) -> np.ndarray:
    """FP proxy filter at S: F = |1+ru|^2 / |1-ru*rd|^2."""
    num = 1.0 + ru_S
    den = 1.0 - (ru_S * rd_S)
    return np.real((np.abs(num) ** 2) / (np.abs(den) ** 2 + 1e-30))


# -----------------------------------------------------------------------------
# PyMoosh integration helpers (centralized)
# -----------------------------------------------------------------------------

def theta_from_u_branch_fixed(n_inc: complex, u: float) -> complex:
    """Convert u=k_parallel/k0 to complex theta with fixed branch.

    Branch policy:
      - choose cos(theta) so that Im(cos(theta)) >= 0
      - then choose sign(theta) so sin(theta) best matches s=u/n

    This keeps kz in the incident medium consistent with Im(kz)>=0 and stabilizes
    TE/TM behavior in evanescent regimes.
    """
    n = n_inc + 0j
    s = (u + 0j) / n
    c = csqrt(1.0 - s * s + 0j)
    if np.imag(c) < 0:
        c = -c
    th = carccos(c)
    if abs(np.sin(th) - s) > abs(np.sin(th) + s):
        th = -th
    return th


def build_structure_for_pymoosh(Structure, incident_n: complex, layers: list[tuple[complex, float]], substrate_n: complex):
    """Build a PyMoosh Structure from complex refractive indices and thicknesses."""
    mats = [incident_n ** 2] + [ni ** 2 for (ni, _di) in layers] + [substrate_n ** 2]
    layer_type = list(range(len(mats)))
    thickness = [0.0] + [float(di) for (_ni, di) in layers] + [0.0]
    return Structure(mats, layer_type, thickness, verbose=False, unit="nm")


def _pymoosh_coeff_r(PyMoosh, st, wavelength_nm: float, theta_rad: complex, pol: int = 0) -> complex:
    """Return complex reflection amplitude from a prebuilt Structure."""
    if hasattr(PyMoosh, "coefficient_S"):
        r, _t, _R, _T = PyMoosh.coefficient_S(st, float(wavelength_nm), theta_rad, int(pol))
    else:
        r, _t, _R, _T = PyMoosh.coefficient(st, float(wavelength_nm), theta_rad, int(pol))
    return complex(r)


def pymoosh_r_spectrum_from_structure(PyMoosh, st, lam_nm: np.ndarray, theta_rad: complex, pol: int = 0) -> np.ndarray:
    """Complex reflection spectrum for an existing Structure."""
    lam = np.asarray(lam_nm, float)
    return np.array([_pymoosh_coeff_r(PyMoosh, st, float(w), theta_rad, pol=pol) for w in lam], dtype=complex)


def pymoosh_stack_reflection(
    PyMoosh,
    Structure,
    incident_n: complex,
    layers: list[tuple[complex, float]],
    substrate_n: complex,
    lam_nm: np.ndarray,
    u: float = 0.0,
    pol: int = 0,
) -> np.ndarray:
    """Complex reflection spectrum for a generic stack using fixed-branch u->theta."""
    st = build_structure_for_pymoosh(Structure, incident_n, layers, substrate_n)
    theta = theta_from_u_branch_fixed(incident_n, float(u))
    return pymoosh_r_spectrum_from_structure(PyMoosh, st, lam_nm, theta, pol=pol)


def pymoosh_stack_reflection_scalar(
    PyMoosh,
    Structure,
    incident_n: complex,
    layers: list[tuple[complex, float]],
    substrate_n: complex,
    wavelength_nm: float,
    u: float = 0.0,
    pol: int = 0,
) -> complex:
    """Scalar-wavelength wrapper for pymoosh_stack_reflection."""
    return complex(
        pymoosh_stack_reflection(
            PyMoosh, Structure, incident_n, layers, substrate_n,
            np.array([float(wavelength_nm)], float), u=u, pol=pol,
        )[0]
    )


def apply_bottom_phase_policy(rb_raw: np.ndarray, phi_b_mode: str | float = "pi") -> np.ndarray:
    """Apply bottom-boundary phase policy to rb_raw.

    Parameters
    ----------
    rb_raw:
        Raw bottom reflection at pHTL|ITO (seen from pHTL), array over lambda.
    phi_b_mode:
        - "pi"  : force phase pi, i.e. rb = -|rb_raw|
        - "raw" : keep raw complex reflection
        - float  : force phase to the given radians while preserving |rb_raw|

    Returns
    -------
    rb_used: complex ndarray
    """
    rb = np.asarray(rb_raw, complex)

    if isinstance(phi_b_mode, str):
        mode = phi_b_mode.strip().lower()
        if mode == "pi":
            return -np.abs(rb).astype(complex)
        if mode in ("raw", "none"):
            return rb.copy()
        raise ValueError("phi_b_mode must be 'pi', 'raw', or a float(rad)")

    phi = float(phi_b_mode)
    return np.abs(rb) * np.exp(1j * phi)


def build_bt_structures(Structure, n0: dict, d: dict):
    """Build standard B/T terminal structures used in this project."""
    st_bottom = build_structure_for_pymoosh(
        Structure,
        n0["pHTL"],
        [(n0["ITO"], d["ITO"]), (n0["anode"], d["anode"])],
        n0["substrate"],
    )
    st_top = build_structure_for_pymoosh(
        Structure,
        n0["ETL"],
        [(n0["cathode1"], d["cathode1"]), (n0["cathode"], d["cathode"]), (n0["CPL"], d["CPL"])],
        n0["air"],
    )
    return st_bottom, st_top


def terminal_reflections_BT_from_structures(
    PyMoosh,
    n0: dict,
    lam_nm: np.ndarray,
    u: float,
    pol: int,
    st_bottom,
    st_top,
    phi_b_mode: str | float = "pi",
    tm_top_pi_shift: bool = False,
    tm_bottom_pi_shift: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (rb_used, rt, rb_raw) for the standard B/T terminal stacks.

    rb_raw and rt are evaluated by PyMoosh at fixed-branch angles for each side,
    then rb_used is produced by apply_bottom_phase_policy.

    Parameters
    ----------
    tm_top_pi_shift:
        If True and pol==1(TM), apply pi-shift at T terminal reflection:
        r_T(TM) -> -r_T(TM).
    tm_bottom_pi_shift:
        If True and pol==1(TM), apply pi-shift at B raw terminal reflection:
        r_B,raw(TM) -> -r_B,raw(TM).

    Notes
    -----
    - With phi_b_mode='pi', B boundary is finally forced to r_B = -|r_B,raw| for both
      TE/TM. Therefore tm_bottom_pi_shift typically does not change rb_used; it is
      kept as an explicit switch for debugging/completeness.
    """
    theta_b = theta_from_u_branch_fixed(n0["pHTL"], float(u))
    theta_t = theta_from_u_branch_fixed(n0["ETL"], float(u))

    rb_raw = pymoosh_r_spectrum_from_structure(PyMoosh, st_bottom, lam_nm, theta_b, pol=pol)
    rt = pymoosh_r_spectrum_from_structure(PyMoosh, st_top, lam_nm, theta_t, pol=pol)

    if int(pol) == 1:
        # TM-only terminal phase policy (independent B/T switches)
        if bool(tm_bottom_pi_shift):
            rb_raw = -rb_raw
        if bool(tm_top_pi_shift):
            rt = -rt

    rb_used = apply_bottom_phase_policy(rb_raw, phi_b_mode=phi_b_mode)
    return rb_used, rt, rb_raw


def terminal_reflections_BT_from_pymoosh(
    PyMoosh,
    Structure,
    n0: dict,
    d: dict,
    lam_nm: np.ndarray,
    u: float,
    pol: int,
    phi_b_mode: str | float = "pi",
    tm_top_pi_shift: bool = False,
    tm_bottom_pi_shift: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: build standard B/T structures then evaluate terminal reflections."""
    st_bottom, st_top = build_bt_structures(Structure, n0, d)
    return terminal_reflections_BT_from_structures(
        PyMoosh=PyMoosh,
        n0=n0,
        lam_nm=lam_nm,
        u=u,
        pol=pol,
        st_bottom=st_bottom,
        st_top=st_top,
        phi_b_mode=phi_b_mode,
        tm_top_pi_shift=tm_top_pi_shift,
        tm_bottom_pi_shift=tm_bottom_pi_shift,
    )


# -----------------------------------------------------------------------------
# Phase3p1: 2D-Green indicators (Eq.54 / Eq.61 / Eq.66)
# -----------------------------------------------------------------------------

def _stabilized_D(ru_S: np.ndarray, rd_S: np.ndarray, eta: float = 0.0) -> np.ndarray:
    """Return stabilized Fabry--Perot denominator D = 1 - ru*rd + i*eta."""
    ru = np.asarray(ru_S, complex)
    rd = np.asarray(rd_S, complex)
    return (1.0 - ru * rd) + 1j * float(eta)


def green_terms_from_ru_rd(ru_S: np.ndarray, rd_S: np.ndarray, eta: float = 0.0) -> dict[str, np.ndarray]:
    r"""Compute phase3p1 Green-based indicators from (ru, rd) at source plane S.

    Definitions (matching phase3_Green_3d.pdf):

      Eq.(54):
        F_sigma^(G)(lambda,u) = |1+ru|^2 |1+rd|^2 / |D|^2

      Eq.(61):
        F_sigma(lambda) = F_sigma^(G)(lambda, u=0)
        (same formula, evaluated at u=0 in caller)

      Eq.(66):
        I_norm(z;lambda,u) = |E(z;lambda,u)|^2 / max_{z in EML}|E(z;lambda,u)|^2
        Here we provide the unnormalized numerator proxy |E|^2 ~ Eq.(54) and
        normalization is handled by normalize_e2_eq66().

    Parameters
    ----------
    ru_S, rd_S:
        Complex effective reflection coefficients at source plane S.
    eta:
        Optional small positive regularizer for D -> D + i*eta.

    Returns
    -------
    dict with keys:
      D, inv_abs_D2, N_eq54_num, F_eq54, F_eq61_u0, E2_raw
    """
    ru = np.asarray(ru_S, complex)
    rd = np.asarray(rd_S, complex)
    D = _stabilized_D(ru, rd, eta=eta)

    absD2 = np.abs(D) ** 2 + 1e-30
    N = (np.abs(1.0 + ru) ** 2) * (np.abs(1.0 + rd) ** 2)

    F_eq54 = np.real(N / absD2)
    # Eq.61 is the same expression at u=0 (handled by caller). Keep alias.
    F_eq61 = F_eq54.copy()
    # Use the same raw magnitude as |E|^2 proxy before Eq.66 normalization.
    E2_raw = F_eq54.copy()

    return {
        "D": D,
        "inv_abs_D2": 1.0 / absD2,
        "N_eq54_num": N,
        "F_eq54": F_eq54,
        "F_eq61_u0": F_eq61,
        "E2_raw": E2_raw,
    }


def green_F_eq54_from_ru_rd(ru_S: np.ndarray, rd_S: np.ndarray, eta: float = 0.0) -> np.ndarray:
    """Eq.(54): F_sigma^(G)(lambda,u) from ru/rd at S."""
    return green_terms_from_ru_rd(ru_S, rd_S, eta=eta)["F_eq54"]


def green_F_eq61_from_ru_rd_u0(ru_S_u0: np.ndarray, rd_S_u0: np.ndarray, eta: float = 0.0) -> np.ndarray:
    """Eq.(61): F_sigma(lambda) at u=0 (same closed form as Eq.54)."""
    return green_terms_from_ru_rd(ru_S_u0, rd_S_u0, eta=eta)["F_eq61_u0"]


def normalize_e2_eq66(E2_z: np.ndarray, eml_mask: np.ndarray | None = None) -> np.ndarray:
    r"""Eq.(66) normalization for |E|^2 profile.

      I_norm(z;lambda,u) = |E(z;lambda,u)|^2 / max_{z in EML}|E(z;lambda,u)|^2

    Parameters
    ----------
    E2_z:
        Nonnegative profile (raw |E|^2 proxy) along z.
    eml_mask:
        Optional boolean mask selecting EML region for normalization maximum.
        If None, normalize by the global maximum.
    """
    arr = np.asarray(E2_z, float)
    if eml_mask is not None:
        m = np.asarray(eml_mask, bool)
        if m.shape != arr.shape:
            raise ValueError("eml_mask must have the same shape as E2_z")
        denom = float(np.max(arr[m])) if np.any(m) else float(np.max(arr))
    else:
        denom = float(np.max(arr))
    return arr / (denom + 1e-30)


def blend_te_tm(F_te: np.ndarray, F_tm: np.ndarray, w_te: float = 0.5, w_tm: float = 0.5) -> np.ndarray:
    """Weighted TE/TM blend used for scalar F(lambda) reporting."""
    return float(w_te) * np.asarray(F_te, float) + float(w_tm) * np.asarray(F_tm, float)


__all__ = [
    # rewrap core
    "k0_of_lambda_nm",
    "kz_nm",
    "wrap_reflection_from_terminal",
    "effective_ru_rd_at_S_from_BT",
    "fp_filter_from_ru_rd",
    # phase3p1 Green indicators
    "green_terms_from_ru_rd",
    "green_F_eq54_from_ru_rd",
    "green_F_eq61_from_ru_rd_u0",
    "normalize_e2_eq66",
    "blend_te_tm",
    # stable terminal reflection (production)
    "stack_reflection_smatrix_spectrum",
    "stack_reflection_smatrix_scalar",
    "terminal_reflections_BT",
    # PyMoosh integration helpers (tests / optional comparisons)
    "theta_from_u_branch_fixed",
    "build_structure_for_pymoosh",
    "pymoosh_r_spectrum_from_structure",
    "pymoosh_stack_reflection",
    "pymoosh_stack_reflection_scalar",
    "apply_bottom_phase_policy",
    "build_bt_structures",
    "terminal_reflections_BT_from_structures",
    "terminal_reflections_BT_from_pymoosh",
]
