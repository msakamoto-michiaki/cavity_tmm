"""Layer and interface matrices (Eq.(1) and Eq.(2))."""

from __future__ import annotations

import numpy as np

from .fresnel import fresnel_rt
from .kz import kz as kz_layer


def layer_matrix(n: complex, n_e: complex, u: float, lambda0_m: float, d_m: float) -> np.ndarray:
    """Eq.(1) layer propagation matrix L^j(d).

    Numerical stability note:
    With the causal branch choice Im(kz) >= 0, the factor exp(-i*kz*d) contains
    exp(+Im(kz)*d), which can overflow inside a transfer-matrix product.
    Since reflection coefficient extraction uses ratios of transfer-matrix
    elements (e.g. -M01/M00), the overall matrix is only defined up to a scalar
    factor. We therefore remove the scalar exp(+Im(kz)*d) from this layer matrix.

    We return:
        L = exp(-Im(kz)*d) * diag(exp(-i*kz*d), exp(+i*kz*d))
          = diag(exp(-i*Re(kz)*d), exp(+i*Re(kz)*d) * exp(-2*Im(kz)*d))

    This keeps magnitudes <= 1 and avoids overflow while preserving r,t.
    """
    kz_j = kz_layer(n, n_e, u, lambda0_m)
    phase = float(np.real(kz_j)) * float(d_m)
    alpha = float(np.imag(kz_j)) * float(d_m)  # alpha >= 0 by kz() branch choice

    e11 = np.exp(-1j * phase)
    e22 = np.exp(1j * phase) * np.exp(-2.0 * alpha)
    return np.array([[e11, 0.0 + 0.0j], [0.0 + 0.0j, e22]], dtype=complex)


def interface_matrix(n_j: complex, n_k: complex, n_e: complex, u: float, lambda0_m: float, pol: str) -> np.ndarray:
    """Eq.(2) interface matrix I^{j/(j+1)}.

    I = (1/t) * [[1, r],[r,1]]
    """
    rt = fresnel_rt(n_j=n_j, n_k=n_k, n_e=n_e, u=u, lambda0_m=lambda0_m, pol=pol)
    r, t = rt.r, rt.t
    return (1.0 / t) * np.array([[1.0 + 0.0j, r], [r, 1.0 + 0.0j]], dtype=complex)
