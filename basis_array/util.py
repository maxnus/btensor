import numpy as np


__all__ = [
        'IdentityMatrix',
        'BasisError',
        'ndot',
        'overlap',
        ]


class IdentityMatrix:
    """Represents the identity matrix of shape size x size."""

    def __init__(self, size=None):
        self.size = size

    def as_array(self):
        return np.identity(self.size)

    @property
    def T(self):
        return self


class BasisError(Exception):
    pass


def ndot(*args):
    args = [x for x in args if not isinstance(x, IdentityMatrix)]
    return np.linalg.multi_dot(args)


def overlap(a, b):
    if a is None and b is None:
        return IdentityMatrix()

    if  a is None or b is None:
        raise BasisError

    return (a | b)
