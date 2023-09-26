#     Copyright 2023 Max Nusspickel
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from __future__ import annotations
from typing import *

import numpy as np
from numpy.typing import ArrayLike

import btensor
from btensor.util import ndot, IdentityMatrix
from btensor.exceptions import BasisError
from btensor.basis import _is_nobasis

if TYPE_CHECKING:
    from numbers import Number
    from btensor import Tensor


def _to_tensor(*args):
    args = tuple(btensor.Tensor(a) if isinstance(a, np.ndarray) else a for a in args)
    if len(args) == 1:
        return args[0]
    return args


def _empty_factory(numpy_func):
    def func(basis, *args, shape=None, **kwargs):
        if shape is None:
            if any(_is_nobasis(b) for b in basis):
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


def _sum(a: ArrayLike | Tensor, axis: int | Tuple[int, ...] | None = None,
         out: np.ndarray | None = None) -> Tensor | Number:
    a = _to_tensor(a)
    value = a.to_numpy(copy=False).sum(axis=axis, out=out)
    if value.ndim == 0:
        return value
    if isinstance(axis, (int, np.integer)):
        axis = (axis,)
    basis, variance = zip(*((a.basis[ax], a.variance[ax]) for ax in range(a.ndim) if ax not in axis))
    return type(a)(value, basis=basis, variance=variance, mode=a.mode, copy_data=False)


def dot(a: ArrayLike | Tensor, b: ArrayLike | Tensor) -> Tensor | Number:
    """Evaluates the dot product of two tensors while performing required basis transformations automatically.

    See also numpy.dot.

    Args:
        a, b: First and second tensor.

    Returns:
        Dot product of a and b.
    """
    a, b = _to_tensor(a, b)
    basis = variance = None
    if a.ndim == b.ndim == 1:
        leftaxis, rightaxis = 0, 0
    elif a.ndim == b.ndim == 2:
        leftaxis, rightaxis = -1, 0
        basis = (a.basis[0], b.basis[1])
        variance = (a.variance[0], b.variance[1])
    elif b.ndim == 1:
        leftaxis, rightaxis = -1, 0
        basis = a.basis[:-1]
        variance = a.variance[:-1]
    elif b.ndim >= 2:
        leftaxis, rightaxis = -1, -2
        basis = a.basis[:-1] + b.basis[:-2] + b.basis[-1:]
        variance = a.variance[:-1] + b.variance[:-2] + b.variance[-1:]
    else:
        raise ValueError(f"invalid dimensions: {a.ndim} and {b.ndim}")
    basis_left = a.basis[leftaxis]
    basis_right = b.basis[rightaxis]
    if basis_left is not btensor.nobasis and basis_right is not btensor.nobasis:
        variance_ovlp = (-a.variance[leftaxis], -b.variance[rightaxis])
        ovlp = basis_left.get_transformation(basis_right, variance=variance_ovlp)
    elif basis_left is btensor.nobasis and basis_right is btensor.nobasis:
        size = a.shape[leftaxis]
        if b.shape[rightaxis] != size:
            raise BasisError
        ovlp = IdentityMatrix(size)
    else:
        raise BasisError(f"Cannot evaluate overlap between {a} and {b}")

    value = ndot(a.to_numpy(copy=False), ovlp, b.to_numpy(copy=False))
    if not isinstance(value, np.ndarray):
        return value
    mode = 'tensor' if (a.mode == 'tensor') or (b.mode == 'tensor') else 'array'
    return type(a)(value, basis=basis, variance=variance, mode=mode, copy_data=False)


def trace(a: ArrayLike | Tensor, axis1: int = 0, axis2: int = 1) -> Tensor | Number:
    a = _to_tensor(a)
    basis1 = a.basis[axis1]
    basis2 = a.basis[axis2]
    if basis1.root != basis2.root:
        raise BasisError("cannot take trace over axis with incompatible bases!")
    if basis1 != basis2:
        parent = basis1.find_common_parent(basis2)
        a = a.change_basis_at(parent, axis=axis1).change_basis_at(parent, axis=axis2)
    if a.variance[axis1] == a.variance[axis2] and not a.basis[axis1].is_orthonormal:
        a = a.change_variance_at(-a.variance[axis2], axis=axis2)
    value = a.to_numpy(copy=False).trace(axis1=axis1, axis2=axis2)
    if value.ndim == 0:
        return value
    if axis1 < 0:
        axis1 += a.ndim
    if axis2 < 0:
        axis2 += a.ndim
    basis_new, variance = zip(*((a.basis[ax], a.variance[ax]) for ax in range(a.ndim) if ax not in {axis1, axis2}))
    return type(a)(value, basis=basis_new, variance=variance, mode=a.mode, copy_data=False)


def moveaxis(a: ArrayLike | Tensor,
             source: int | Sequence[int],
             destination: int | Sequence[int],
             name: str | None = None) -> Tensor:
    a = _to_tensor(a)
    source = np.core.numeric.normalize_axis_tuple(source, a.ndim, 'source')
    destination = np.core.numeric.normalize_axis_tuple(destination, a.ndim, 'destination')
    values_orig = a.to_numpy(copy=False)
    values = np.moveaxis(values_orig, source=source, destination=destination)
    order = [n for n in range(a.ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)
    basis, variance = zip(*((a.basis[ax], a.variance[ax]) for ax in order))
    return type(a)(values, basis=basis, variance=variance, name=name, mode=a.mode)
