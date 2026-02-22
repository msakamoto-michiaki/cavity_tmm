"""System / transfer matrix helpers.

We use 2x2 transfer matrices per polarization (TE/TM).

Convention:
    E^+ : forward wave travelling +z
    E^- : backward wave travelling -z

Transfer matrix M maps amplitudes on the right to the left:

    [E_L^+, E_L^-]^T = M [E_R^+, E_R^-]^T.

This module provides:
1) A generic multilayer transfer matrix builder (interface + propagation).
2) GPVM "system matrices" S^A and S^B (Eq.(6)-(7)) built from the stack
   for a source plane located inside the emission layer.
3) Extraction of reflection/transmission coefficients exactly as Eq.(12).

Notes on Eq.(12):
- For S^B, the usual left-incident extraction applies:
    r_B = s21/s11,  t_B = 1/s11.
- For S^A, Eq.(6) uses the boundary condition E_0^+ = 0 (no incoming wave
  from the left ambient), which corresponds to an incidence from the RIGHT
  onto the left-side stack. In that case:
    r_A = E_a^+ / E_a^- = -s12/s11,
    t_A = det(S^A)/s11.
"""

from __future__ import annotations

import numpy as np

from .matrices import interface_matrix, layer_matrix


def _renorm(M: np.ndarray) -> np.ndarray:
    """Renormalize a 2x2 transfer matrix by a scalar to avoid overflow.

    Scaling M by a nonzero scalar does not change reflection coefficients extracted
    via ratios of elements (e.g. -M01/M00).
    """
    s = float(np.max(np.abs(M)))
    if (not np.isfinite(s)) or (s == 0.0):
        return M
    return M / s


def stack_transfer_matrix(
    n_list: list[complex],
    d_list_m: list[float],
    n_e: complex,
    u: float,
    lambda0_m: float,
    pol: str,
) -> np.ndarray:
    """Build the transfer matrix for a multilayer.

    Parameters
    ----------
    n_list : list of complex
        Refractive indices including left and right semi-infinite media.
        Length = N.
    d_list_m : list of float
        Thicknesses (meters) for the internal layers only:
        corresponds to layers 1..N-2. Length must be N-2.
    n_e, u, lambda0_m, pol : see other modules.

    Returns
    -------
    M : (2,2) complex ndarray
        Transfer matrix mapping right-side amplitudes to left-side.

    Sequence:
        M = I^{0/1} L^1(d1) I^{1/2} L^2(d2) ... I^{N-2/(N-1)}
    """
    if len(n_list) < 2:
        raise ValueError("n_list must have at least two media")
    if len(d_list_m) != max(0, len(n_list) - 2):
        raise ValueError(
            f"d_list_m length must be len(n_list)-2. Got {len(d_list_m)} for len(n_list)={len(n_list)}"
        )

    M = np.eye(2, dtype=complex)

    # Interface 0/1
    M = _renorm(M @ interface_matrix(n_list[0], n_list[1], n_e, u, lambda0_m, pol))

    # For each internal layer j=1..N-2: propagate, then interface j/(j+1)
    for j in range(1, len(n_list) - 1):
        if 1 <= j <= len(n_list) - 2:
            d_m = float(d_list_m[j - 1])
            M = _renorm(M @ layer_matrix(n_list[j], n_e, u, lambda0_m, d_m))
        # interface to next medium
        M = _renorm(M @ interface_matrix(n_list[j], n_list[j + 1], n_e, u, lambda0_m, pol))

    return M


def rt_from_transfer_matrix(M: np.ndarray) -> tuple[complex, complex]:
    """Return (r,t) for incidence from the LEFT.

    With boundary condition E_R^- = 0 (no incoming from the right):
        r = E_L^- / E_L^+ = M21/M11
        t = E_R^+ / E_L^+ = 1/M11

    This matches Eq.(12) for S^B.
    """
    M = np.asarray(M, dtype=complex)
    if M.shape != (2, 2):
        raise ValueError("M must be 2x2")
    if abs(M[0, 0]) == 0:
        raise ZeroDivisionError("M[0,0] is zero; cannot extract r,t")
    r = M[1, 0] / M[0, 0]
    t = 1.0 / M[0, 0]
    return complex(r), complex(t)


def rt_incident_from_right(M: np.ndarray) -> tuple[complex, complex]:
    """Return (r_right, t_right) for incidence from the RIGHT.

    Setup:
        - Incident wave on the right traveling toward -z has unit amplitude:
            E_R^- = 1
        - No incoming wave from the left ambient:
            E_L^+ = 0

    Solve [E_L^+, E_L^-]^T = M [E_R^+, E_R^-]^T.

    Then
        r_right = E_R^+ / E_R^- = -M12/M11
        t_right = E_L^- / E_R^- = det(M)/M11

    This matches Eq.(12) for S^A (r_A, t_A).
    """
    M = np.asarray(M, dtype=complex)
    if M.shape != (2, 2):
        raise ValueError("M must be 2x2")
    if abs(M[0, 0]) == 0:
        raise ZeroDivisionError("M[0,0] is zero; cannot extract r,t")
    r = -M[0, 1] / M[0, 0]
    # For 2x2, compute determinant explicitly.
    # np.linalg.det can introduce avoidable numerical error for ill-conditioned
    # complex matrices encountered in deep stacks.
    det = (M[0, 0] * M[1, 1]) - (M[0, 1] * M[1, 0])
    t = det / M[0, 0]
    return complex(r), complex(t)


def build_system_matrices_SA_SB(
    n_list: list[complex],
    d_list_m: list[float],
    eml_index: int,
    z_ex_m: float,
    n_e: complex,
    u: float,
    lambda0_m: float,
    pol: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build GPVM system matrices S^A and S^B (Eq.(6)-(7)).

    n_list includes left and right semi-infinite media (indices 0..N-1).
    d_list_m gives thicknesses of internal layers 1..N-2.

    eml_index = w is the index of the EML inside n_list (1..N-2).
    z_ex_m is measured from the left boundary of the EML (0<z_ex<d_eml).

    Returns (SA, SB) as 2x2 complex matrices.
    """
    N = len(n_list)
    if N < 3:
        raise ValueError("n_list must contain at least 3 media")
    if len(d_list_m) != N - 2:
        raise ValueError("d_list_m must have length len(n_list)-2")
    if not (1 <= eml_index <= N - 2):
        raise ValueError("eml_index must be in [1, N-2]")

    d_eml = float(d_list_m[eml_index - 1])
    if not (0.0 < z_ex_m < d_eml):
        raise ValueError("z_ex_m must satisfy 0 < z_ex_m < d_eml")

    # S^A = I^{0/1} L^1(d1) ... I^{w-1/w} L^w(z_ex)
    SA = np.eye(2, dtype=complex)

    SA = SA @ interface_matrix(n_list[0], n_list[1], n_e, u, lambda0_m, pol)
    for j in range(1, eml_index):
        SA = SA @ layer_matrix(n_list[j], n_e, u, lambda0_m, float(d_list_m[j - 1]))
        SA = SA @ interface_matrix(n_list[j], n_list[j + 1], n_e, u, lambda0_m, pol)
    SA = SA @ layer_matrix(n_list[eml_index], n_e, u, lambda0_m, float(z_ex_m))

    # S^B = L^w(d_eml-z_ex) I^{w/(w+1)} ... L^{N-2}(d) I^{(N-2)/(N-1)}
    SB = np.eye(2, dtype=complex)

    SB = SB @ layer_matrix(n_list[eml_index], n_e, u, lambda0_m, float(d_eml - z_ex_m))
    SB = SB @ interface_matrix(n_list[eml_index], n_list[eml_index + 1], n_e, u, lambda0_m, pol)
    for j in range(eml_index + 1, N - 1):
        if 1 <= j <= N - 2:
            SB = SB @ layer_matrix(n_list[j], n_e, u, lambda0_m, float(d_list_m[j - 1]))
        SB = SB @ interface_matrix(n_list[j], n_list[j + 1], n_e, u, lambda0_m, pol)

    return SA, SB


def eq12_rt_from_SA_SB(SA: np.ndarray, SB: np.ndarray) -> tuple[complex, complex, complex, complex]:
    """Extract (rA, tA, rB, tB) exactly as Eq.(12) from SA/SB."""
    SA = np.asarray(SA, dtype=complex)
    SB = np.asarray(SB, dtype=complex)
    if SA.shape != (2, 2) or SB.shape != (2, 2):
        raise ValueError("SA and SB must be 2x2")
    if abs(SA[0, 0]) == 0 or abs(SB[0, 0]) == 0:
        raise ZeroDivisionError("S11 is zero; cannot extract coefficients")

    rA = -SA[0, 1] / SA[0, 0]
    tA = (SA[0, 0] * SA[1, 1] - SA[0, 1] * SA[1, 0]) / SA[0, 0]

    rB = SB[1, 0] / SB[0, 0]
    tB = 1.0 / SB[0, 0]

    return complex(rA), complex(tA), complex(rB), complex(tB)
