from __future__ import annotations

import numpy as np


class ClampBounds:
    """
    Clamp particle positions to the box constraints.

    If a particle goes outside the box, its position is clipped back into
    the valid range. The velocity component that caused the collision is
    reset to zero for extra stability.
    """

    def __init__(self, lower, upper):
        """
        Parameters
        ----------
        lower : list | np.ndarray
            Lower bounds.
        upper : list | np.ndarray
            Upper bounds.
        """
        self.lower = np.array(lower, dtype=float)
        self.upper = np.array(upper, dtype=float)

    def apply(self, position, velocity):
        """
        Apply clamping to a particle state.

        Parameters
        ----------
        position : np.ndarray
            Current particle position.
        velocity : np.ndarray
            Current particle velocity.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Corrected position and corrected velocity.
        """
        position = np.array(position, dtype=float)
        velocity = np.array(velocity, dtype=float)

        new_position = np.clip(position, self.lower, self.upper)
        new_velocity = velocity.copy()

        # If a coordinate hits a boundary, nullify that velocity component.
        hit_mask = new_position != position
        new_velocity[hit_mask] = 0.0

        return new_position, new_velocity
