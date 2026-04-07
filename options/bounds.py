"""options.bounds

Boundary enforcement strategies for PSO particles.

All strategies implement the BoundsPolicy interface so they can be
swapped transparently inside PSO without changing the core loop.

Strategy implemented
--------------------
ClampBounds (chosen strategy):
    Clips the position to [lower, upper] and zeroes the velocity component
    on any axis that hit a wall. This avoids particles bouncing back and
    forth near boundaries, which can destabilise convergence, especially
    in high dimensions.

    Trade-off: zeroing velocity introduces a mild bias toward the walls,
    but in practice the cognitive/social terms quickly pull the particle
    back into the interior. Alternatives (reflect, penalty) are left as
    extension points via the ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BoundsPolicy(ABC):
    """Abstract interface for boundary enforcement."""

    @abstractmethod
    def apply(
        self, position: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Enforce bounds on a particle state.

        Parameters
        ----------
        position : np.ndarray
        velocity : np.ndarray

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Corrected (position, velocity).
        """
        raise NotImplementedError


class ClampBounds(BoundsPolicy):
    """
    Clamp particle positions to the box constraints.

    If a particle exits the search box, its position is clipped back to
    the boundary and the velocity component that caused the collision is
    reset to zero for extra stability.
    """

    def __init__(self, lower, upper) -> None:
        """
        Parameters
        ----------
        lower : list | np.ndarray   Lower bounds per dimension.
        upper : list | np.ndarray   Upper bounds per dimension.
        """
        self.lower = np.array(lower, dtype=float)
        self.upper = np.array(upper, dtype=float)

    def apply(
        self, position: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply clamping to the position and zero velocity on axes that hit the wall.

        Parameters
        ----------
        position : np.ndarray
            The current position of the particle.
        velocity : np.ndarray
            The current velocity of the particle.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The corrected position and velocity.
        """
        position = np.array(position, dtype=float)
        velocity = np.array(velocity, dtype=float)

        new_position = np.clip(position, self.lower, self.upper)
        new_velocity = velocity.copy()

        # Zero the velocity on axes that hit a wall.
        hit_mask = new_position != position
        new_velocity[hit_mask] = 0.0

        return new_position, new_velocity
