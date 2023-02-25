import numpy as np


__all__ = [
        'IdentityMatrix'
        ]


class IdentityMatrix:
    """Represents the identity matrix of shape size x size."""

    def __init__(self, size):
        self.size = size

    def as_array(self):
        return np.identity(self.size)

    @property
    def T(self):
        return self
