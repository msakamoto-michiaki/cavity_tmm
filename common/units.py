# common/units.py
# -*- coding: utf-8 -*-
"""
Unit conversion helpers (no dependencies).

Policy:
- Keep these functions pure and dependency-free.
- Accept scalars or numpy arrays.
- Always return the same "shape type" as input (scalar -> scalar, array -> array).
"""

from __future__ import annotations

from typing import Union, overload

import numpy as np

Number = Union[int, float, np.number]
ArrayLike = Union[Number, np.ndarray]


@overload
def nm_to_m(x_nm: Number) -> float: ...
@overload
def nm_to_m(x_nm: np.ndarray) -> np.ndarray: ...


def nm_to_m(x_nm: ArrayLike):
    """
    Convert nanometers to meters.

    Parameters
    ----------
    x_nm : float | int | np.number | np.ndarray
        Value(s) in nanometers.

    Returns
    -------
    float | np.ndarray
        Value(s) in meters.

    Notes
    -----
    Uses exact factor 1e-9.
    """
    if np.isscalar(x_nm):
        return float(x_nm) * 1e-9
    x_nm = np.asarray(x_nm)
    return x_nm * 1e-9


@overload
def m_to_nm(x_m: Number) -> float: ...
@overload
def m_to_nm(x_m: np.ndarray) -> np.ndarray: ...


def m_to_nm(x_m: ArrayLike):
    """
    Convert meters to nanometers.

    Parameters
    ----------
    x_m : float | int | np.number | np.ndarray
        Value(s) in meters.

    Returns
    -------
    float | np.ndarray
        Value(s) in nanometers.

    Notes
    -----
    Uses exact factor 1e9.
    """
    if np.isscalar(x_m):
        return float(x_m) * 1e9
    x_m = np.asarray(x_m)
    return x_m * 1e9


@overload
def um_to_m(x_um: Number) -> float: ...
@overload
def um_to_m(x_um: np.ndarray) -> np.ndarray: ...


def um_to_m(x_um: ArrayLike):
    """
    Convert micrometers to meters.
    """
    if np.isscalar(x_um):
        return float(x_um) * 1e-6
    x_um = np.asarray(x_um)
    return x_um * 1e-6


@overload
def m_to_um(x_m: Number) -> float: ...
@overload
def m_to_um(x_m: np.ndarray) -> np.ndarray: ...


def m_to_um(x_m: ArrayLike):
    """
    Convert meters to micrometers.
    """
    if np.isscalar(x_m):
        return float(x_m) * 1e6
    x_m = np.asarray(x_m)
    return x_m * 1e6
