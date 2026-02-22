"""Stub module for PyMoosh.

PyMoosh optionally imports `refractiveindex.RefractiveIndexMaterial` to access
an external dispersion database. This project assumes wavelength-independent
complex refractive indices (n,k constants), so the database is not required.

Keeping this stub prevents an ImportError when PyMoosh is imported.
"""


class RefractiveIndexMaterial:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "RefractiveIndexMaterial is unavailable in this environment. "
            "This project uses constant complex refractive indices instead."
        )
