from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numbers import Number

import numpy as np
from numpy.typing import ArrayLike

import btensor
from btensor.util import ndot, BasisError, IdentityMatrix
from btensor.core.basis import is_nobasis, BasisInterface


def _to_tensor(*args):
    args = tuple(btensor.Tensor(a) if isinstance(a, np.ndarray) else a for a in args)
    if len(args) == 1:
        return args[0]
    return args


def _empty_factory(numpy_func):
    def func(basis, *args, shape=None, **kwargs):
        if shape is None:
            if any(is_nobasis(b) for b in basis):
                raise ValueError("cannot deduce size of nobasis. Specify shape explicitly")
            shape = tuple(b.size for b in basis)
        return btensor.Tensor(numpy_func(shape, *args, **kwargs), basis=basis)
    return func


empty = _empty_factory(np.empty)
ones = _empty_factory(np.ones)
zeros = _empty_factory(np.zeros)


def _empty_like_factory(func):
    def func_like(a, *args, **kwargs):
        a = _to_tensor(a)
        return func(a.basis, *args, shape=a.shape, **kwargs)
    return func_like


zeros_like = _empty_like_factory(zeros)
empty_like = _empty_like_factory(empty)
ones_like = _empty_like_factory(ones)


def _sum(a: ArrayLike | btensor.Tensor, axis=None) -> btensor.Tensor | Number:
    a = _to_tensor(a)
    value = a._data.sum(axis=axis)
    if value.ndim == 0:
        return value
    if axis is None:
        return value
    if isinstance(axis, (int, np.integer)):
        axis = (axis,)
    basis = tuple(a.basis[ax] for ax in range(a.ndim) if ax not in axis)
    return type(a)(value, basis=basis)


def _overlap(a: BasisInterface, b: BasisInterface):
    if is_nobasis(a) and is_nobasis(b):
        return IdentityMatrix(None)
    if is_nobasis(a) or is_nobasis(b):
        raise BasisError(f"Cannot evaluate overlap between {a} and {b}")
    return a.get_overlap(b)


def dot(a: ArrayLike | btensor.Tensor, b: ArrayLike | btensor.Tensor) -> btensor.Tensor | Number:
    a, b = _to_tensor(a, b)
    if a.ndim == b.ndim == 1:
        ovlp = _overlap(a.basis[0], b.basis[0])
        return ndot(a._data, ovlp, b._data)
    if a.ndim == b.ndim == 2:
        ovlp = _overlap(a.basis[-1], b.basis[0])
        out = ndot(a._data, ovlp, b._data)
        basis = (a.basis[0], b.basis[1])
    elif b.ndim == 1:
        ovlp = _overlap(a.basis[-1], b.basis[0])
        out = ndot(a._data, ovlp, b._data)
        basis = a.basis[:-1]
    elif b.ndim >= 2:
        ovlp = _overlap(a.basis[-1], b.basis[-2])
        out = ndot(a._data, ovlp, b._data)
        basis = (a.basis[:-1] + b.basis[:-2] + b.basis[-1:])
    return type(a)(out, basis=basis)


def trace(a: ArrayLike | btensor.Tensor, axis1: int = 0, axis2: int = 1) -> btensor.Tensor | Number:
    a = _to_tensor(a)
    basis1 = a.basis[axis1]
    basis2 = a.basis[axis2]
    if basis1.root != basis2.root:
        raise BasisError("Cannot take trace over axis with incompatible bases!")
    if basis1 == basis2:
        a = a
    else:
        parent = basis1.find_common_parent(basis2)
        a = a.change_basis_at(axis1, parent).change_basis_at(axis2, parent)
    value = a._data.trace(axis1=axis1, axis2=axis2)
    if value.ndim == 0:
        return value
    if axis1 < 0:
        axis1 += a.ndim
    if axis2 < 0:
        axis2 += a.ndim
    basis_new = tuple(a.basis[i] for i in set(range(a.ndim)) - {axis1, axis2})
    return type(a)(value, basis_new)


