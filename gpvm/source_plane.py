"""Solve the field amplitudes at source-plane boundaries (GPVM Sec.2.1).

This module solves for the four plane-wave amplitudes at the *immediate* left/right
side of the source plane (dipole sheet):

    (E_a^+, E_a^-)  : right/left-going waves just on the left side of the source plane
    (E_b^+, E_b^-)  : right/left-going waves just on the right side of the source plane

They obey the source-plane jump condition (paper Eq.(4)):

    [E_a^+; E_a^-] + [A^+; A^-] = [E_b^+; E_b^-].

IMPORTANT (convention used in this codebase)
-------------------------------------------
The reflection coefficients rA, rB expected by :func:`solve_source_plane_fields`
are the *effective* reflection coefficients referenced **at the source plane**,
consistent with how this repository builds the system matrices (Eq.(6)-(7)):

    [0; E_0^-]      = S^A [E_a^+; E_a^-]
    [E_b^+; E_b^-]  = S^B [E_{n+1}^+; 0]

and extracts

    rA = -S^A_12 / S^A_11,
    rB =  S^B_21 / S^B_11.

With this convention, the algebra is the *phase-free* form obtained by rewriting
paper Eq.(8)-(11) in terms of source-plane-referenced reflections.

If you instead define rA, rB at the EML interfaces (planes A/B) and keep the
explicit exp(i k_z 2 z_ex) factors, do NOT use this solver as-is; either convert
those interface-referenced reflections to source-plane-referenced reflections,
or implement the paper's Eq.(8)-(11) literally.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SourcePlaneFields:
    Ea_plus: complex
    Ea_minus: complex
    Eb_plus: complex
    Eb_minus: complex


def solve_source_plane_fields(
    A_plus: complex,
    A_minus: complex,
    rA: complex,
    rB: complex,
    kz_e: complex,
    d_e_m: float,
    z_ex_m: float,
) -> SourcePlaneFields:
    """Compute (E_a^+, E_a^-, E_b^+, E_b^-) from rA, rB at the source plane.

    Parameters
    ----------
    A_plus, A_minus : complex
        Source terms (Eq.(5)).
    rA, rB : complex
        Effective back (A-side) and front (B-side) reflection coefficients
        referenced at the source plane (see module docstring).
    kz_e, d_e_m, z_ex_m
        Accepted for API compatibility, but not used by the phase-free algebra.
    """
    A_plus = complex(A_plus)
    A_minus = complex(A_minus)
    rA = complex(rA)
    rB = complex(rB)
    # NOTE: The following is equivalent to the paper's Eq.(8)-(11) after
    # converting interface-referenced reflections to source-plane-referenced ones:
    #   rA_src = rA_int * exp(i k_z 2 z_ex)
    #   rB_src = rB_int * exp(i k_z 2 (d_e - z_ex))
    # so that (rA_int rB_int exp(i k_z 2 d_e)) == (rA_src rB_src).
    denom = 1.0 - (rA * rB)

    Ea_minus = (A_plus * rB - A_minus) / denom
    Ea_plus = rA * Ea_minus
    Eb_plus = Ea_plus + A_plus
    Eb_minus = Ea_minus + A_minus

    return SourcePlaneFields(
        Ea_plus=complex(Ea_plus),
        Ea_minus=complex(Ea_minus),
        Eb_plus=complex(Eb_plus),
        Eb_minus=complex(Eb_minus),
    )


def check_eq4(fields: SourcePlaneFields, A_plus: complex, A_minus: complex, atol: float = 1e-12) -> bool:
    """Check Eq.(4): [Ea+;Ea-] + [A+;A-] = [Eb+;Eb-]."""
    lhs_plus = fields.Ea_plus + complex(A_plus)
    lhs_minus = fields.Ea_minus + complex(A_minus)
    return (abs(lhs_plus - fields.Eb_plus) <= atol) and (abs(lhs_minus - fields.Eb_minus) <= atol)
