"""Fresnel reflection/transmission coefficients in the GPVM convention.

Implements Eq. (3) of the GPVM paper (Optics Express 27(16), 2019).

Note: The paper uses a Fresnel convention where TM reflection has an
additional minus sign compared to some common optics textbooks. We follow
the paper exactly so that the GPVM matrices (Eq. (2)) match.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .kz import kz as kz_layer


@dataclass(frozen=True)
class FresnelRT:
    r: complex
    t: complex


def fresnel_rt(n_j: complex, n_k: complex, n_e: complex, u: float, lambda0_m: float, pol: str) -> FresnelRT:
    """Return (r,t) at the interface from layer j to layer k.

    Parameters
    ----------
    n_j, n_k : complex
        Refractive indices of incident and transmitted media.
    n_e : complex
        EML refractive index (used to parameterize k_parallel).
    u : float
        Normalized in-plane wavevector.
    lambda0_m : float
        Vacuum wavelength.
    pol : {'TE','TM'}

    Returns
    -------
    FresnelRT
    """
    pol_u = pol.upper()
    kz_j = kz_layer(n_j, n_e, u, lambda0_m)
    kz_k = kz_layer(n_k, n_e, u, lambda0_m)

    if pol_u == 'TE':
        r = (kz_j - kz_k) / (kz_j + kz_k)
        t = (2.0 * kz_j) / (kz_j + kz_k)
        return FresnelRT(r=complex(r), t=complex(t))

    if pol_u == 'TM':
        # Eq.(3): r_TM = -(kz_j/n_j^2 - kz_k/n_k^2)/(kz_j/n_j^2 + kz_k/n_k^2)
        r = -((kz_j / (complex(n_j) ** 2)) - (kz_k / (complex(n_k) ** 2))) / (
            (kz_j / (complex(n_j) ** 2)) + (kz_k / (complex(n_k) ** 2))
        )
        # Eq.(3): t_TM = 2*kz_j / (kz_j*(n_k/n_j) + kz_k*(n_j/n_k))
        denom = kz_j * (complex(n_k) / complex(n_j)) + kz_k * (complex(n_j) / complex(n_k))
        t = (2.0 * kz_j) / denom
        return FresnelRT(r=complex(r), t=complex(t))

    raise ValueError(f"pol must be 'TE' or 'TM', got {pol!r}")
