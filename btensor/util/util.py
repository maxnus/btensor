from contextlib import contextmanager
import numpy as np
from .matrix import Matrix, IdentityMatrix


__all__ = [
        'is_int',
        'array_like',
        'atleast_1d',
        'BasisError',
        'ndot',
        #'overlap',
        'expand_axis',
        'replace_attr',
        ]



def is_int(obj):
    return isinstance(obj, (int, np.integer))


def array_like(obj):
    try:
        obj.shape
        obj.ndim
        obj[()]
        return True
    except (AttributeError, TypeError):
        return False


def atleast_1d(obj):
    return tuple(np.atleast_1d(obj))


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


@contextmanager
def replace_attr(obj, **kwargs):
    """Temporary replace attributes and methods of object."""
    orig = {}
    try:
        for name, attr in kwargs.items():
            orig[name] = getattr(obj, name)
            # For functions: replace and bind as method, otherwise just set
            setattr(obj, name, attr.__get__(obj) if callable(attr) else attr)
        yield obj
    finally:
        # Restore originals
        for name, attr in orig.items():
            setattr(obj, name, attr.__get__(obj) if callable(attr) else attr)
