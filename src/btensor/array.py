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
from numbers import Number
import operator

import numpy as np

from btensor.util import *
from btensor.exceptions import BasisDependentOperationError
from btensor.basis import Basis, _Variance, NBasis
from btensor.tensor import Tensor, DOCSTRING_TEMPLATE
from btensor import numpy_functions

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class Array(Tensor):

    __doc__ = \
        """A numerical container class with support for automatic basis transformation.
        """ + DOCSTRING_TEMPLATE.format(name="tensor", default_variance=_Variance.CONTRAVARIANT)


    def __getitem__(self, key):
        """Construct and return sub-Array."""

        # getitem of Tensor base class:
        try:
            return super().__getitem__(key)
        except IndexError as e:
            pass

        if isinstance(key, int):
            return type(self)(self._data[key], basis=self.basis[1:])
        if isinstance(key, (list, np.ndarray)):
            value = self._data[key]
            basis = (self.basis[0].make_basis(key),) + self.basis[1:]
            return type(self)(value, basis=basis)
        if isinstance(key, slice) or key is np.newaxis:
            key = (key,)
        if isinstance(key, tuple):
            value = self._data[key]
            if value.ndim == 0:
                return value

            # Add nobasis for each newaxis (None) key
            newaxis_indices = [i for (i, k) in enumerate(key) if (k is np.newaxis)]
            basis = list(self.basis)
            for i in newaxis_indices:
                basis.insert(i, 1)

            # Replace Ellipsis with multiple slice(None)
            if Ellipsis in key:
                idx = key.index(Ellipsis)
                ellipsis_size = len(basis) - len(key) + 1
                key = key[:idx] + ellipsis_size*(slice(None),) + key[idx+1:]

            for i, ki in enumerate(reversed(key), start=1):
                idx = len(key) - i
                if isinstance(ki, (int, np.integer)):
                    del basis[idx]
                elif isinstance(ki, slice):
                    basis[idx] = Basis(argument=ki, parent=basis[idx])
                elif ki is np.newaxis:
                    pass
                else:
                    raise ValueError("key %r of type %r" % (ki, type(ki)))
            basis = tuple(basis)
            return type(self)(value, basis=basis)
        raise NotImplementedError("Key= %r of type %r" % (key, type(key)))

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value._data
        with replace_attr(self._data.flags, writeable=True):
            self._data[key] = value
        # Not required, since np.newaxis has no effect in assignment?
        #if not isinstance(key, tuple) or np.newaxis not in key:
        #    return
        #basis_old = list(self.basis)
        #basis_new = tuple(nobasis if elem is np.newaxis else basis_old.pop(0) for elem in key)
        #self.basis = basis_new

    def to_tensor(self) -> Tensor:
        return Tensor(self._data, basis=self.basis)

    def sum(self, axis: int | Tuple[int] | None = None) -> Array | Number:
        return numpy_functions.sum(self, axis=axis)

    @property
    def __array_interface__(self):
        return self._data.__array_interface__

    def __mul__(self, other: Number | Array) -> Array:
        if not isinstance(other, Number) and self.basis != other.basis:
            raise BasisDependentOperationError
        return self._operator(operator.mul, other)

    def __truediv__(self, other: Number | Array) -> Array:
        if not isinstance(other, Number) and self.basis != other.basis:
            raise BasisDependentOperationError
        return self._operator(operator.truediv, other)

    def _operator_check_allowed(self, op: Callable, other: Number | Array | None = None,
                                reverse: bool = False) -> Self:
        if other is None:
            return self._operator(op, reverse=reverse)
        if isinstance(other, Tensor):
            raise BasisDependentOperationError("operation only allowed between two arrays, not array and tensor")
        if not isinstance(other, Number) and self.basis != other.basis:
            raise BasisDependentOperationError("operation only allowed for arrays with the same basis")
        return self._operator(op, other, reverse=reverse)

    def __floordiv__(self, other: Number | Array) -> Self:
        return self._operator_check_allowed(operator.floordiv, other)

    def __mod__(self, other: Number | Array) -> Self:
        return self._operator_check_allowed(operator.mod, other)

    def __pow__(self, other: Number | Array) -> Self:
        return self._operator_check_allowed(operator.pow, other)

    def __rtruediv__(self, other: Number | Array) -> Self:
        return self._operator_check_allowed(operator.truediv, other, reverse=True)

    def __rfloordiv__(self, other: Number | Array) -> Self:
        return self._operator_check_allowed(operator.floordiv, other, reverse=True)

    def __rpow__(self, other: Number | Array) -> Self:
        return self._operator_check_allowed(operator.pow, other, reverse=True)

    def __gt__(self, other: Number | Array) -> Self:
        return self._operator_check_allowed(operator.gt, other)

    def __ge__(self, other: Number | Array) -> Self:
        return self._operator_check_allowed(operator.ge, other)

    def __lt__(self, other: Number | Array) -> Self:
        return self._operator_check_allowed(operator.lt, other)

    def __le__(self, other: Number | Array) -> Self:
        return self._operator_check_allowed(operator.le, other)

    def __abs__(self) -> Self:
        return self._operator(operator.abs)


def Coarray(data: ArrayLike,
            basis: NBasis | None = None,
            variance: Sequence[int] | None = None,
            name: str | None = None,
            copy_data: bool = True) -> Tensor:
    data = np.array(data, copy=copy_data)
    if variance is None:
        variance = data.ndim * [_Variance.COVARIANT]
    return Array(data, basis=basis, variance=variance, name=name, copy_data=False)


Coarray.__doc__ = \
    """A helper function to create arrays with default variance 1 (covariant).
    """ + DOCSTRING_TEMPLATE.format(name=Coarray.__name__.lower(), default_variance=_Variance.COVARIANT)
