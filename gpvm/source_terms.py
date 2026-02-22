"""Dipole source terms (Eq.(5))."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SourceTerms:
    A_plus: complex
    A_minus: complex


def source_terms(u: float, pol: str, orientation: str) -> SourceTerms:
    """Return A^{+}, A^{-} for the dipole source at the source plane.

    Implements Eq.(5) of the GPVM paper.

    Parameters
    ----------
    u : float
        Normalized in-plane wavevector.
    pol : {'TE','TM'}
    orientation : {'h','v'}
        Dipole orientation: horizontal (h) or vertical (v).

    Notes
    -----
    - For TE: only horizontal dipole contributes (vertical is zero).
    - For horizontal dipole (both TE and TM): A_plus = -A0, A_minus = +A0.
    - For vertical dipole (TM only): A_plus = A_minus = Av.
    """
    pol_u = pol.upper()
    ori = orientation.lower()

    if ori not in ('h', 'v'):
        raise ValueError("orientation must be 'h' or 'v'")

    if pol_u == 'TE':
        if ori == 'v':
            return SourceTerms(0.0 + 0.0j, 0.0 + 0.0j)
        A0 = np.sqrt(3.0 / (16.0 * np.pi))
        return SourceTerms(A_plus=complex(-A0), A_minus=complex(+A0))

    if pol_u == 'TM':
        if ori == 'h':
            A0 = np.sqrt(3.0 / (16.0 * np.pi)) * np.sqrt(max(0.0, 1.0 - float(u) ** 2))
            return SourceTerms(A_plus=complex(-A0), A_minus=complex(+A0))
        # vertical
        Av = np.sqrt(3.0 / (8.0 * np.pi)) * float(u)
        return SourceTerms(A_plus=complex(Av), A_minus=complex(Av))

    raise ValueError("pol must be 'TE' or 'TM'")
