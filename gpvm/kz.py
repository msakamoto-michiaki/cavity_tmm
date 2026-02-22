"""kz utilities.

In the GPVM paper, the in-plane wavevector is parameterized by

    k_parallel = n_e * k0 * u,

where n_e is the (lossless) refractive index of the EML and u is the
normalized in-plane wavevector.

Then for any layer j,

    kz_j = k0 * sqrt(n_j^2 - n_e^2 * u^2).

We enforce the decaying/causal branch for evanescent waves by requiring
Im(kz) >= 0. For purely real kz, we keep kz >= 0.
"""

from __future__ import annotations

import numpy as np


def k0_from_lambda(lambda0_m: float) -> float:
    """Vacuum wavenumber 2*pi/lambda."""
    return 2.0 * np.pi / float(lambda0_m)


def kz(n: complex, n_e: complex, u: float, lambda0_m: float) -> complex:
    """Compute kz in layer with refractive index n.

    Parameters
    ----------
    n : complex
        Refractive index of the layer.
    n_e : complex
        Refractive index of the (lossless) EML used for normalization.
    u : float
        Normalized in-plane wavevector.
    lambda0_m : float
        Wavelength in vacuum (meters).

    Returns
    -------
    kz : complex
        z-component of wavevector.
    """
    k0 = k0_from_lambda(lambda0_m)
    # Use complex dtype to avoid NaNs for negative real arguments.
    arg = (complex(n) ** 2) - (complex(n_e) ** 2) * (float(u) ** 2)
    root = np.lib.scimath.sqrt(arg)
    kz_val = k0 * root

    # Enforce Im(kz) >= 0 (evanescent decay away from interface).
    if kz_val.imag < 0:
        kz_val = -kz_val
    # For purely real values, enforce kz >= 0.
    if abs(kz_val.imag) < 1e-15 and kz_val.real < 0:
        kz_val = -kz_val
    return complex(kz_val)
