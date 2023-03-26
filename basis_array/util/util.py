import numpy as np
from .matrix import Matrix, IdentityMatrix


__all__ = [
        'nobasis',
        'BasisError',
        'ndot',
        'overlap',
        ]


nobasis = type('NoBasis', (object,), {})()


class BasisError(Exception):
    pass


def ndot(*args):
    args = [x for x in args if not isinstance(x, IdentityMatrix)]
    return np.linalg.multi_dot(args)


def overlap(a, b):
    if a is nobasis and b is nobasis:
        return IdentityMatrix()
    if a is nobasis or b is nobasis:
        raise BasisError
    return (a | b)
