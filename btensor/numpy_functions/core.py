import numpy as np
import btensor as bt
from btensor.util import ndot, overlap
from btensor.util import BasisError


__all__ = [
    'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like', 'sum', 'dot', 'trace',
]


def _to_tensor(*args):
    args = tuple(bt.Tensor(a, basis=a.ndim*(bt.nobasis,)) if isinstance(a, np.ndarray) else a for a in args)
    return args[0] if len(args) == 1 else args


def _empty_factory(numpy_func):
    def func(basis, *args, **kwargs):
        shape = tuple(b.size if isinstance(b, bt.Basis) else b for b in basis)
        basis = tuple(b if isinstance(b, bt.Basis) else bt.nobasis for b in basis)
        return bt.Tensor(numpy_func(shape, *args, **kwargs), basis=basis)
    return func


empty = _empty_factory(np.empty)
ones = _empty_factory(np.ones)
zeros = _empty_factory(np.zeros)


def _empty_like_factory(func):
    def func_like(a, *args, **kwargs):
        a = _to_tensor(a)
        basis = tuple((b if not (b is bt.nobasis) else n) for b, n in zip(a.basis, a.shape))
        return func(basis, *args, **kwargs)
    return func_like


zeros_like = _empty_like_factory(zeros)
empty_like = _empty_like_factory(empty)
ones_like = _empty_like_factory(ones)


def sum(a, axis=None):
    a = _to_tensor(a)
    value = a._value.sum(axis=axis)
    if value.ndim == 0:
        return value
    if axis is None:
        return value
    if isinstance(axis, (int, np.integer)):
        axis = (axis,)
    basis = tuple(a.basis[ax] for ax in range(a.ndim) if ax not in axis)
    return type(a)(value, basis=basis)


def dot(a, b):
    a, b = _to_tensor(a, b)
    if a.ndim == b.ndim == 1:
        ovlp = overlap(a.basis[0], b.basis[0])
        return ndot(a._value, ovlp, b._value)
    if a.ndim == b.ndim == 2:
        ovlp = overlap(a.basis[-1], b.basis[0])
        out = ndot(a._value, ovlp, b._value)
        basis = (a.basis[0], b.basis[1])
    elif b.ndim == 1:
        ovlp = overlap(a.basis[-1], b.basis[0])
        out = ndot(a._value, ovlp, b._value)
        basis = a.basis[:-1]
    elif b.ndim >= 2:
        ovlp = overlap(a.basis[-1], b.basis[-2])
        out = ndot(a._value, ovlp, b._value)
        basis = (a.basis[:-1] + b.basis[:-2] + b.basis[-1:])
    return type(a)(out, basis=basis)


def trace(a, axis1=0, axis2=1):
    a = _to_tensor(a)
    basis1 = a.basis[axis1]
    basis2 = a.basis[axis2]
    if basis1.root != basis2.root:
        raise BasisError("Cannot take trace over axis with incompatible bases!")
    if basis1 == basis2:
        a = a
    else:
        parent = basis1.find_common_parent(basis2)
        a = a.as_basis_at(axis1, parent).as_basis_at(axis2, parent)
    value = a._value.trace(axis1=axis1, axis2=axis2)
    if value.ndim == 0:
        return value
    if axis1 < 0:
        axis1 += a.ndim
    if axis2 < 0:
        axis2 += a.ndim
    basis_new = tuple(a.basis[i] for i in set(range(a.ndim)) - {axis1, axis2})
    return type(a)(value, basis_new)
