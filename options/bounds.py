import numpy as np


class ClampBounds:
    def __init__(self, lower, upper):
        self.lower = np.array(lower)
        self.upper = np.array(upper)

    def apply(self, position, velocity):
        new_position = np.clip(position, self.lower, self.upper)
        return new_position, velocity
