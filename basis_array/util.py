import numpy as np


__all__ = [
        'nobasis',
        'IdentityMatrix',
        'BasisError',
        'ndot',
        'overlap',
        ]


nobasis = type('NoBasis', (object,), {})()


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
    if a is nobasis and b is nobasis:
        return IdentityMatrix()

    if  a is nobasis or b is nobasis:
        raise BasisError

    return (a | b)
