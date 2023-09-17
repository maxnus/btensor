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
from btensor.basis import Basis, _is_basis_or_nobasis, _is_nobasis, compatible_basis, nobasis, IBasis, NBasis, _Variance
from btensor.basistuple import BasisTuple
from btensor import numpy_functions


class _ChangeBasisInterface:

    def __init__(self, tensor: Tensor) -> None:
        self._tensor = tensor

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._tensor})"

    def __getitem__(self, key: NBasis) -> Tensor:
        return self._tensor.change_basis(key)


DOCSTRING_TEMPLATE = \
    """A numerical container class with support for automatic basis transformation.

    Args:
        data: NumPy array containing the representation of the {name}.
        basis: Basis object or tuple of Basis objects, representing the Basis along each axis of the input data.
            Default: nobasis along each axis.
        variance: Variance along each dimension. Default: {default_variance}.
        name: Name of the {name}. Default: 'Basis<ID>' where <ID> is the ID of the basis.
        copy_data: If False, no copy of the NumPy data will be created. Default: True.

    """


class Tensor:

    __doc__ = DOCSTRING_TEMPLATE.format(name="Tensor", default_variance=_Variance.CONTRAVARIANT)
    _SUPPORTED_DTYPE = [np.int8, np.int16, np.int32, np.int64,
                        np.float16, np.float32, np.float64]

    def __init__(self,
                 data: ArrayLike,
                 basis: NBasis | None = None,
                 variance: Sequence[int] | None = None,
                 name: str | None = None,
                 copy_data: bool = True) -> None:
        """Create new Tensor instance."""
        data = np.array(data, copy=copy_data)
        if data.dtype not in self._SUPPORTED_DTYPE:
            raise ValueError(f"dtype {data.dtype} is not supported")
        self._data = data
        self.name = name
        if basis is None:
            basis = data.ndim * (nobasis,)
        if variance is None:
            variance = data.ndim * [_Variance.CONTRAVARIANT]
        variance = tuple(variance)
        basis = BasisTuple.create(basis)
        self._basis = basis
        self._variance = variance
        self._check_basis(basis)
        self._cob = _ChangeBasisInterface(self)

    def __repr__(self) -> str:
        basis_names = f"({', '.join([bas.name for bas in self.basis]) + (',' if self.ndim == 1 else '')})"
        attrs = dict(basis=basis_names, variance=self.variance, dtype=self.dtype)
        if self.name is not None:
            attrs['name'] = self.name
        attrs = ', '.join([f"{key}= {val}" for (key, val) in attrs.items()])
        return f'{type(self).__name__}({attrs})'

    def copy(self, name: str | None = None, copy_data: bool = True) -> Tensor:
        """Create a copy of the tensor.

        Args:
            name: Name of the copied tensor.
            copy_data: If True, a copy of the underlying NumPy data will be performed, otherwise the copied tensor will
                refer to the same data. Default: True.

        Returns:
            Copy of tensor.

        """
        return type(self)(self._data, basis=self.basis, variance=self.variance, name=name, copy_data=copy_data)

    # --- Basis

    @property
    def basis(self) -> BasisTuple[IBasis, ...]:
        """Basis of the tensor."""
        return self._basis

    def _check_basis(self, basis: BasisTuple) -> None:
        if len(basis) != self.ndim:
            raise ValueError(f"{self.ndim}-dimensional Array requires {self.ndim} basis elements ({len(basis)} given)")
        for axis, (size, baselem) in enumerate(zip(self.shape, basis)):
            if not _is_basis_or_nobasis(baselem):
                raise ValueError(f"basis instance or nobasis required (given: {baselem} of type {type(baselem)}).")
            if _is_nobasis(baselem):
                continue
            if size != baselem.size:
                raise ValueError(f"axis {axis} with size {size} incompatible with basis size {baselem.size}")

    def replace_basis(self, basis: tuple[IBasis | None, ...], inplace: bool = False) -> Tensor:
        """Replace basis of tensor with a new basis.

        Args:
            basis: New basis.
            inplace: If True, the tensor will be modified in-place, otherwise a new Tensor instance will be created.
                Default: False.

        Returns:
            Tensor with replaced basis.

        """
        tensor = self if inplace else self.copy()
        new_basis = self.basis.update_with(basis)
        tensor._check_basis(new_basis)
        tensor._basis = new_basis
        return tensor

    # --- Variance

    @property
    def variance(self) -> tuple[int, ...]:
        """Tuple with `ndim` elements with value 1 or -1, for covariance and contravariance, respectively."""
        return self._variance

    def _check_variance(self, variance: Sequence[int]) -> None:
        if len(variance) != self.ndim:
            raise ValueError(f"{self.ndim}-dimensional Array requires {self.ndim} variance elements "
                             f"({len(variance)} given)")
        for var in variance:
            if var not in {-1, 1}:
                raise ValueError(f"variance can only contain elements -1 and 1 (not {var})")

    def change_variance(self, variance: Sequence[int]) -> Tensor:
        """Change variance of tensor and transform representation accordingly.

        Args:
            variance: Sequence of new variances for tensor along each axis, -1 for contravariance and 1 for covariance.

        Returns:
            Tensor with variance changed.

        """
        self._check_variance(variance)
        changed_axes = [i for (i, v, w) in enumerate(zip(self.variance, variance)) if v != w]
        result = self
        for axis, var in zip(changed_axes, variance):
            result = result.change_variance_at(var, axis=axis)
        return result

    def change_variance_at(self, variance: int, axis: int):
        """Change variance of tensor along a single axis and transform representation accordingly.

        Args:
            variance: New variance for tensor along axis, -1 for contravariance and 1 for covariance.
            axis: Axis along which the variance is changed.

        Returns:
            Tensor with variance changed.

        """
        if variance not in {-1, 1}:
            raise ValueError(f"variance can only be -1 and 1 (not {variance})")
        if self.variance[axis] == variance:
            return self
        if self.basis[axis].is_orthonormal:
            values = self._data
        else:
            labels_in = string.ascii_lowercase[:self.ndim]
            labels_out = list(labels_in)
            labels_out[axis] = 'Z'
            labels_out = ''.join(labels_out)
            contraction = f"{labels_in},{labels_in[axis]}{labels_out[axis]}->{labels_out}"
            # to lower index:
            metric = self.basis[axis].metric
            # to raise index:
            if variance == -1:
                metric = metric.inverse
            values = np.einsum(contraction, self._data, metric.to_numpy())
        variance_tuple = self.variance[:axis] + (variance,) + self.variance[axis+1:]
        return type(self)(values, basis=self.basis, variance=variance_tuple)

    @staticmethod
    def _get_basis_transform(basis1: NBasis, basis2: NBasis, variance: tuple[int, int]):
        return basis1._get_overlap_mpl(basis2, variance=variance, simplify=True)

    def __getitem__(self, key: slice | Ellipsis | NBasis) -> Tensor:
        if (isinstance(key, slice) and key == slice(None)) or key is Ellipsis:
            return self
        if isinstance(key, Basis):
            key = (key,)

        type_error_msg = (f'only instances of Basis, slice(None), and Ellipsis are valid indices for the'
                          f'{type(self).__name__} class. '
                          f'Use Array class to index using integers, slices, and integer or boolean arrays')
        if not isinstance(key, tuple):
            raise TypeError(type_error_msg)
        for bas in key:
            if not (isinstance(bas, Basis) or bas in (slice(None), Ellipsis)):
                raise TypeError(type_error_msg)

        return self.project(key)

    def project(self, basis: NBasis) -> Tensor:
        """Transforms tensor to a different set of basis.

        Slice(None) can be used to indicate no transformation.
        Note that this can reduce the rank of the array, for example when projecting onto an orthogonal space.

        Args:
            basis: New basis.

        Returns:
            Tensor projected onto `basis`.

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

    def change_basis(self, basis: NBasis) -> Tensor:
        """Change basis of tensor.

        Slice(None) can be used to indicate no transformation.
        In contrast to `project`, this function will first test if the tensor can be fully represented in the new basis
        and raise a `BasisError` otherwise.

        Args:
            basis: New basis of tensor.

        Returns:
            Tensor in new basis.

        """
        basis = BasisTuple.create_from_default(basis, default=self.basis)
        if not basis.is_spanning(self._basis):
            raise BasisError(f"{basis} does not span {self.basis}")
        return self.project(basis)

    @property
    def cob(self) -> _ChangeBasisInterface:
        """Change of basis interface, similar to `change_basis`, but using []-syntax.

        Slice(None) can be used to indicate no transformation.

        """
        return self._cob

    def change_basis_at(self, basis: IBasis | Sequence[IBasis], axis: int | Sequence[int]) -> Tensor:
        """Change basis of tensor along one or more selected axes.

        Slice(None) can be used to indicate no transformation.
        In contrast to `project`, this function will first test if the tensor can be fully represented in the new basis
        and raise a `BasisError` otherwise.

        Args:
            basis: New basis of tensor for the specified axis.
            axis: One or more axes, along which the basis will be replaced.

        Returns:
            Tensor in new basis.

        """
        # Recursive implementation for sequence arguments:
        if is_sequence(axis):
            t = self.change_basis_at(basis[0], axis[0])
            return t.change_basis_at(basis[1:], axis[1:])

        if axis < 0:
            axis += self.ndim
        basis_new = self.basis[:axis] + (basis,) + self.basis[axis + 1:]
        return self.change_basis(basis_new)

    # Arithmetic

    def is_compatible(self, other: Tensor) -> bool:
        """Check if the basis of the tensor is compatible with another tensor's basis along each axis.

        Args:
            other: The second tensor.

        Returns:
            True if the bases are compatible along each axis, False otherwise.

        """
        return all(self._compatible_axes(other))

    def _compatible_axes(self, other: Tensor) -> list[bool]:
        """Returns a boolean array, indicating for each axis whether the bases are compatible."""
        axes = []
        for i, (b1, b2) in enumerate(zip(self.basis, other.basis)):
            axes.append(bool(compatible_basis(b1, b2)))
        if self.ndim > other.ndim:
            axes += (self.ndim-other.ndim)*[False]
        return axes

    def get_common_basis(self, other: Tensor) -> BasisTuple:
        """Get tuple of common bases between two tensors along each axis.

        Args:
            other: The second tensor.

        Returns:
            A tuple, where each element is the common basis between the two tensors along the respective axis.

        """
        return self.basis.get_common_basistuple(other.basis)

    # --- NumPy compatibility

    @property
    def dtype(self) -> np.dtype:
        """NumPy's dtype of the underlying ndarray."""
        return self._data.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions of the tensor."""
        return self._data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the tensor."""
        return self._data.shape

    def to_numpy(self,
                 basis: NBasis | None = None,
                 copy: bool = True) -> np.ndarray:
        """Get representation of tensor as a NumPy ndarray.

        Args:
            basis: Basis used for the representation. Default: None, which uses the current basis.
            copy: If True, a copy is created of the NumPy data. Default: True.

        Returns:
            NumPy array representation of the tensor in the specified basis.

        """
        if basis is None:
            if copy:
                return self._data.copy()
            return self._data
        return self.change_basis(basis=basis)._data

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        """Return a tensor with axes transposed.

        Args:
            axes: If specified, it must be a tuple or list which contains a permutation
                of [0,1,...,N-1] where N is the number of axes of the tensor. The `i`'th axis
                of the returned tensor will correspond to the axis numbered ``axes[i]``
                of the input. If not specified, defaults to ``range(a.ndim)[::-1]``,
                which reverses the order of the axes.

        Returns:
            Tensor with it axes permuted.

        """
        value = self._data.transpose(axes)
        if axes is None:
            basis = self.basis[::-1]
        else:
            basis = tuple(self.basis[ax] for ax in axes)
        return type(self)(value, basis=basis)

    @property
    def T(self) -> Tensor:
        """Transposed tensor.

        Same as ``self.transpose()``.

        """
        return self.transpose()

    def trace(self, axis1: int = 0, axis2: int = 1) -> Tensor | Number:
        """Returns the sum along diagonals of the tensor.

        Args:
            axis1, axis2 :
                Axes to be used as the first and second axis of the 2-D sub-tensors from which the
                diagonals should be taken. Defaults are the first two axes of the tensor.

        Returns:
            If the tensor is 2-D, the sum along along the diagonal is returned. If the tensor is `N`-D,
            with `N > 2`, then a `(N-2)-D` tensor of sums along diagonals is returned.

        """
        return numpy_functions.trace(self, axis1=axis1, axis2=axis2)

    def dot(self, other: Tensor | np.ndarray) -> Tensor:
        """Dot product of two tensors.

        Args:
            other: The second tensor.

        Returns:
            A tensor representing the result of the dot product.

        """
        return numpy_functions.dot(self, other)

    def _operator(self, op: Callable, *other: Number | Tensor, swap: bool = False) -> Tensor:
        # Unary operator
        if len(other) == 0:
            return type(self)(op(self._data), basis=self.basis)
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
                basis = self.get_common_basis(other)
                v1 = self.change_basis(basis)._data
                v2 = other.change_basis(basis)._data
            else:
                raise ValueError(f"{self} and {other} are not compatible")
        else:
            return NotImplemented
        if swap:
            v1, v2 = v2, v1
        return type(self)(op(v1, v2), basis=basis)

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


def Cotensor(data: ArrayLike,
             basis: NBasis | None = None,
             variance: Sequence[int] | None = None,
             name: str | None = None,
             copy_data: bool = True) -> Tensor:
    data = np.array(data, copy=copy_data)
    if variance is None:
        variance = data.ndim * [_Variance.COVARIANT]
    return Tensor(data, basis=basis, variance=variance, name=name, copy_data=False)


Cotensor.__doc__ = DOCSTRING_TEMPLATE.format(name=Cotensor.__name__, default_variance=_Variance.COVARIANT)
