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
from numbers import Number
import string
import operator
from typing import *

import numpy as np
from numpy.typing import ArrayLike

from btensor.util import *
from btensor.basis import Basis, _is_basis, _is_nobasis, compatible_basis, nobasis, BasisInterface, BasisT
from .basistuple import BasisTuple
from btensor import numpy_functions


class _ChangeBasisInterface:

    def __init__(self, tensor: Tensor) -> None:
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.tensor})"

    def __getitem__(self, key: BasisT) -> Tensor:
        return self.tensor.change_basis(key)


class Tensor:
    """A numerical container class with support for automatic basis transformation."""

    _SUPPORTED_DTYPE = [np.int8, np.int16, np.int32, np.int64,
                        np.float16, np.float32, np.float64]
    _DEFAULT_VARIANCE = -1

    def __init__(self,
                 data: ArrayLike,
                 basis: BasisT | None = None,
                 variance: Sequence[int] | None = None,
                 name: str | None = None,
                 copy_data: bool = True) -> None:
        data = np.array(data, copy=copy_data)
        if data.dtype not in self._SUPPORTED_DTYPE:
            raise ValueError(f"dtype {data.dtype} is not supported")
        #data.flags.writeable = False
        self._data = data
        self.name = name
        if basis is None:
            basis = data.ndim * (nobasis,)
        if variance is None:
            variance = data.ndim * [self._DEFAULT_VARIANCE]
        variance = tuple(variance)
        basis = BasisTuple.create(basis)
        self._basis = basis
        self._variance = variance
        self.check_basis(basis)
        self._cob = _ChangeBasisInterface(self)

    def __repr__(self) -> str:
        attrs = dict(shape=self.shape, dtype=self.dtype)
        if any(np.asarray(self.variance) != self._DEFAULT_VARIANCE):
            attrs['variance'] = self.variance
        if self.name is not None:
            attrs['name'] = self.name
        attrs = ', '.join([f"{key}= {val}" for (key, val) in attrs.items()])
        return f'{type(self).__name__}({attrs})'

    def copy(self, name: str | None = None, copy_data: bool = True) -> Tensor:
        """Create a copy of the tensor.

        Parameters
        ----------
        name :
            Name of the copy.
        copy_data :
            If True, a copy of the underlying NumPy data will be performed, otherwise the copied tensor will refer
            to the same data.

        Returns
        -------
        tensor :
            Copy of tensor
        """
        return type(self)(self._data, basis=self.basis, variance=self.variance, name=name, copy_data=copy_data)

    # --- Basis

    @property
    def basis(self) -> BasisTuple[BasisInterface, ...]:
        return self._basis

    def check_basis(self, basis: BasisTuple) -> None:
        if len(basis) != self.ndim:
            raise ValueError(f"{self.ndim}-dimensional Array requires {self.ndim} basis elements ({len(basis)} given)")
        for axis, (size, baselem) in enumerate(zip(self.shape, basis)):
            if not _is_basis(baselem):
                raise ValueError(f"Basis instance or nobasis required (given: {baselem} of type {type(baselem)}).")
            if _is_nobasis(baselem):
                continue
            if size != baselem.size:
                raise ValueError(f"axis {axis} with size {size} incompatible with basis size {baselem.size}")

    def replace_basis(self, basis: tuple[BasisInterface | None, ...], inplace: bool = False) -> Tensor:
        """Replace basis with new basis."""
        tensor = self if inplace else self.copy()
        new_basis = self.basis.update_with(basis)
        tensor.check_basis(new_basis)
        tensor._basis = new_basis
        return tensor

    # --- Variance

    @property
    def variance(self) -> tuple[int, ...]:
        return self._variance

    #def as_variance(self, variance):
    #    if np.ndim(variance) == 0:
    #        variance = self.ndim * (variance,)
    #    if len(variance) != self.ndim:
    #        raise ValueError(f"{self.ndim}-dimensional Array requires {self.ndim} variance elements "
    #                         f"({len(variance)} given)")
    #    if not np.isin(variance, (-1, 1)):
    #        raise ValueError("Variance can only contain values -1 and 1")
    #    new_basis = []
    #    for i, (b, v0, v1) in enumerate(zip(self.basis, self.variance, variance)):
    #        if v0 != v1:
    #            b = b.dual()
    #        new_basis.append(b)
    #    return self.as_basis(basis=new_basis)

    @property
    def variance_string(self) -> str:
        """String representation of variance tuple."""
        symbols = {1: '+', -1: '-', 0: '*'}
        return ''.join(symbols[x] for x in self.variance)

    @staticmethod
    def _get_basis_transform(basis1: BasisT, basis2: BasisT, variance: tuple[int, int]):
        return basis1._get_overlap_mpl(basis2, variance=variance, simplify=True)

    def __getitem__(self, key: slice | Ellipsis | BasisT) -> Tensor:
        if (isinstance(key, slice) and key == slice(None)) or key is Ellipsis:
            return self
        if isinstance(key, Basis):
            key = (key,)

        type_error = TypeError(f'only instances of Basis, slice(None), and Ellipsis are valid indices for the'
                               f'{type(self).__name__} class. '
                               f'Use Array class to index using integers, slices, and integer or boolean arrays')

        if not isinstance(key, tuple):
            raise type_error
        for bas in key:
            if not (isinstance(bas, Basis) or bas in (slice(None), Ellipsis)):
                raise type_error

        return self.project(key)

    def project(self, basis: BasisT) -> Tensor:
        """Transform to different set of basis.

        Slice(None) can be used to indicate no transformation.

        Note that this can reduce the rank of the array, for example when trying to express
        a purely occupied quantitiy in a purely virtual basis.
        """
        basis = BasisTuple.create_from_default(basis, default=self.basis)
        if basis == self.basis:
            return self

        subscripts = string.ascii_lowercase[:self.ndim]
        operands = [self._data]
        result = list(subscripts)
        basis_out = list(self.basis)
        for i, (bas_curr, bas_out, var, sub) in enumerate(zip(self.basis, basis, self.variance, subscripts)):
            if bas_out == bas_curr:
                continue
            basis_out[i] = bas_out
            # Remove or add basis:
            if _is_nobasis(bas_curr) or _is_nobasis(bas_out):
                continue

            # Avoid evaluating the overlap, if not necessary (e.g. for a permutation matrix)
            ovlp = bas_curr._get_overlap_mpl(bas_out, variance=(-var, var)).simplify()
            if len(ovlp) == 1 and isinstance(ovlp[0], IdentityMatrix):
                raise NotImplementedError
            elif len(ovlp) == 1 and isinstance(ovlp[0], PermutationMatrix):
                perm = ovlp[0]
                indices = perm.indices
                if isinstance(perm, RowPermutationMatrix):
                    operands[0] = util.expand_axis(operands[0], perm.shape[1], indices=indices, axis=i)
                if isinstance(perm, ColumnPermutationMatrix):
                    operands[0] = np.take(operands[0], indices, axis=i)
                continue
            else:
                ovlp = ovlp.evaluate()
            operands.append(ovlp)
            subnew = sub.upper()
            subscripts += f',{sub}{subnew}'
            result[i] = subnew

        basis_out = tuple(basis_out)
        subscripts += '->' + (''.join(result))
        value = np.einsum(subscripts, *operands, optimize=True)
        return type(self)(value, basis=basis_out)

    # --- Change of basis

    def change_basis(self, basis: BasisT) -> Tensor:
        basis = BasisTuple.create_from_default(basis, default=self.basis)
        if not basis.is_spanning(self._basis):
            raise BasisError(f"{basis} does not span {self.basis}")
        return self.project(basis)

    @property
    def cob(self) -> _ChangeBasisInterface:
        return self._cob

    def change_basis_at(self, index: int, basis: BasisInterface) -> Tensor:
        if index < 0:
            index += self.ndim
        basis_new = self.basis[:index] + (basis,) + self.basis[index+1:]
        return self.change_basis(basis_new)

    def __or__(self, basis: BasisT) -> Tensor:
        """To allow basis transformation as (array | basis)"""
        # Left-pad with slice(None), such that the basis transformation applies to the last n axes
        basis = BasisTuple.create_from_default(basis, default=self.basis, leftpad=True)
        return self.change_basis(basis)

    def __ror__(self, basis: BasisT) -> Tensor:
        """To allow basis transformation as (basis | array)"""
        basis = BasisTuple.create_from_default(basis, default=self.basis)
        return self.change_basis(basis)

    # Arithmetic

    def is_compatible(self, other: Tensor) -> bool:
        return all(self.compatible_axes(other))

    def compatible_axes(self, other: Tensor) -> list[int]:
        axes = []
        for i, (b1, b2) in enumerate(zip(self.basis, other.basis)):
            axes.append(bool(compatible_basis(b1, b2)))
        if self.ndim > other.ndim:
            axes += (self.ndim-other.ndim)*[False]
        return axes

    def common_basis(self, other: Tensor) -> BasisTuple:
        return self.basis.get_common_basistuple(other.basis)

    # --- NumPy compatibility

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def to_numpy(self,
                 basis: BasisT | None = None,
                 project: bool = False,
                 copy: bool = True) -> np.ndarray:
        """Convert to NumPy ndarray"""
        if basis is not None:
            transform = self.project if project else self.change_basis
            tensor = transform(basis=basis)
        else:
            tensor = self
        nparray = tensor._data
        if copy:
            return nparray.copy()
        return nparray

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        value = self._data.transpose(axes)
        if axes is None:
            basis = self.basis[::-1]
        else:
            basis = tuple(self.basis[ax] for ax in axes)
        return type(self)(value, basis=basis)

    @property
    def T(self) -> Tensor:
        """Transpose tensor."""
        return self.transpose()

    def trace(self, axis1: int = 0, axis2: int = 1) -> Tensor | Number:
        return numpy_functions.trace(self, axis1=axis1, axis2=axis2)

    def dot(self, other: Tensor | np.ndarray) -> Tensor:
        return numpy_functions.dot(self, other)

    def _operator(self, operator, *other: Number | Tensor, swap: bool = False) -> Tensor:
        # Unary operator
        if len(other) == 0:
            return type(self)(operator(self._data), basis=self.basis)
        # Ternary+ operator
        if len(other) > 1:
            raise NotImplementedError
        # Binary operator
        other = other[0]
        basis = self.basis
        v1 = self._data
        if isinstance(other, Number):
            v2 = other
        elif isinstance(other, Tensor):
            if self.basis == other.basis:
                v2 = other._data
            elif self.is_compatible(other):
                basis = self.common_basis(other)
                v1 = self.change_basis(basis)._data
                v2 = other.change_basis(basis)._data
            else:
                raise ValueError(f"{self} and {other} are not compatible")
        else:
            return NotImplemented
        if swap:
            v1, v2 = v2, v1
        return type(self)(operator(v1, v2), basis=basis)

    # Fully supported:

    def __add__(self, other: Number | Tensor) -> Tensor:
        return self._operator(operator.add, other)

    def __sub__(self, other: Number | Tensor) -> Tensor:
        return self._operator(operator.sub, other)

    def __radd__(self, other: Number | Tensor) -> Tensor:
        return self._operator(operator.add, other, swap=True)

    def __rsub__(self, other: Number | Tensor) -> Tensor:
        return self._operator(operator.sub, other, swap=True)

    def __rmul__(self, other: Number | Tensor) -> Tensor:
        return self._operator(operator.mul, other, swap=True)

    def __pos__(self) -> Tensor:
        return self._operator(operator.pos)

    def __neg__(self) -> Tensor:
        return self._operator(operator.neg)

    def __eq__(self, other):
        return self._operator(operator.eq, other)

    def __ne__(self, other):
        return self._operator(operator.ne, other)

    # Only supported for numbers:

    def __mul__(self, other: Number) -> Tensor:
        if not isinstance(other, Number):
            return NotImplemented
        return self._operator(operator.mul, other)

    def __truediv__(self, other: Number) -> Tensor:
        if not isinstance(other, Number):
            return NotImplemented
        return self._operator(operator.truediv, other)

    # Not allowed due to basis dependence:

    def sum(self, *args: Never, **kwargs: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __floordiv__(self, other: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __mod__(self, other: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __pow__(self, other: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __rtruediv__(self, other: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __rfloordiv__(self, other: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __rpow__(self, other: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __gt__(self, other: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __ge__(self, other: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __lt__(self, other: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __le__(self, other: Never) -> NoReturn:
        raise BasisDependentOperationError

    def __abs__(self) -> NoReturn:
        raise BasisDependentOperationError


class Cotensor(Tensor):

    _DEFAULT_VARIANCE = 1
