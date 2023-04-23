import numpy as np
from .matrix import Matrix, IdentityMatrix


__all__ = [
        'array_like',
        'nobasis',
        'BasisError',
        'ndot',
        'overlap',
        'expand_axis',
        ]


nobasis = type('NoBasis', (object,), {})()


def array_like(obj):
    try:
        obj.shape
        obj.ndim
        obj[()]
    except (AttributeError, TypeError):
        return False
    return True


class BasisError(Exception):
    pass


def ndot(*args):
    args = [x for x in args if not isinstance(x, IdentityMatrix)]
    return np.linalg.multi_dot(args)


def overlap(a, b):
    if a is nobasis and b is nobasis:
        return IdentityMatrix(None)
    if a is nobasis or b is nobasis:
        raise BasisError
    return (a | b)


def expand_axis(a, size, indices=None, axis=-1):
    """Expand NumPy array along axis."""
    shape = list(a.shape)
    shape[axis] = size
    if indices is None:
        indices = slice(a.shape[axis])
    if len(np.arange(size)[indices]) != a.shape[axis]:
        raise ValueError
    if axis < 0:
        axis += a.ndim
    mask = axis * (slice(None),) + (indices,) + (a.ndim - axis - 1) * (slice(None),)
    b = np.zeros_like(a, shape=shape)
    b[mask] = a
    return b