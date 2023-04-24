import numpy as np
import btensor as bt
from btensor.util import ndot, overlap
from btensor.util import BasisError


__all__ = [
    'zeros', 'sum', 'dot', 'trace',
]

def zeros(basis, **kwargs):
    shape = tuple(b.size for b in basis)
    return bt.Tensor(np.zeros(shape, **kwargs), basis=basis)


def sum(a, axis=None):
    value = a._value.sum(axis=axis)
    if value.ndim == 0:
        return value
    if axis is None:
        return value
    if isinstance(axis, (int, np.integer)):
        axis = (axis,)
    basis = [a.basis[ax] for ax in range(a.ndim) if ax not in axis]
    return type(a)(value, basis=basis)


def dot(a, b):
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
    basis_new = tuple(a.basis[i] for i in set(range(a.ndim)) - set((axis1, axis2)))
    return type(a)(value, basis_new)
