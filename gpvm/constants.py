"""Physical constants (SI).

Kept minimal on purpose. Extend as needed.
"""

import numpy as np

# Speed of light in vacuum [m/s]
C0: float = 299_792_458.0

# Vacuum permeability [H/m]
MU0: float = 4e-7 * np.pi

# Vacuum permittivity [F/m]
EPS0: float = 1.0 / (MU0 * C0 * C0)
