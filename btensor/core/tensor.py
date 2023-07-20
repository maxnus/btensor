from __future__ import annotations
import numbers
import string
from typing import *

import numpy as np
import scipy
from numpy.typing import ArrayLike

from btensor.util import *
from .basis import Basis, is_basis, is_nobasis, compatible_basis, nobasis, BasisInterface, TBasis
from .basistuple import BasisTuple
from .optemplate import OperatorTemplate
from btensor import numpy_functions


class Tensor(OperatorTemplate):

    DEFAULT_VARIANCE = -1

    def __init__(self,
                 data: ArrayLike,
                 basis: TBasis | None= None,
                 variance: Sequence[int] | None = None,
                 copy_data: bool = True) -> None:
        data = np.array(data, copy=copy_data)
        data.flags.writeable = False
        self._data = data
        if basis is None:
            basis = data.ndim * (nobasis,)
        if variance is None:
            variance = data.ndim * [self.DEFAULT_VARIANCE]
        variance = tuple(variance)

        basis = BasisTuple.create(basis)
        self._basis = basis
        self._variance = variance
        self.check_basis(basis)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(shape= {self.shape}, variance= {self.variance})'

    def copy(self) -> Tensor:
        return type(self)(self._data, basis=self.basis, copy_data=True)

    # --- Basis

    @property
    def basis(self) -> BasisTuple[BasisInterface, ...]:
        return self._basis

    def check_basis(self, basis: BasisTuple) -> None:
        if len(basis) != self.ndim:
            raise ValueError(f"{self.ndim}-dimensional Array requires {self.ndim} basis elements ({len(basis)} given)")
        for axis, (size, baselem) in enumerate(zip(self.shape, basis)):
            if not is_basis(baselem):
                raise ValueError(f"Basis instance or nobasis required (given: {baselem} of type {type(baselem)}).")
            if is_nobasis(baselem):
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

    # ---

    @staticmethod
    def _get_basis_transform(basis1: TBasis, basis2: TBasis, variance: tuple[int, int]):
        return basis1._get_overlap_mpl(basis2, variance=variance, simplify=True)

    def __getitem__(self, key: slice | Ellipsis | TBasis) -> Tensor:
        if (isinstance(key, slice) and key == slice(None)) or key is Ellipsis:
            return self
        if isinstance(key, Basis):
            key = (key,)

        index_error = IndexError(f'only instances of Basis, slice(None), and Ellipsis are valid indices for the'
                                 f'{type(self).__name__} class. '
                                 f'Use Array class to index using integers, slices, and integer or boolean arrays')

        if not isinstance(key, tuple):
            raise index_error
        for bas in key:
            if not isinstance(bas, Basis) or key in (slice(None), Ellipsis):
                raise index_error

        return self.project(key)

    def project(self, basis: TBasis) -> Tensor:
        """Transform to different set of basis.

        Slice(None) can be used to indicate no transformation.

        Note that this can reduce the rank of the array, for example when trying to express
        a purely occupied quantitiy in a purely virtual basis.
        """
        basis = BasisTuple.create_from_default(basis, default=self.basis)

        subscripts = string.ascii_lowercase[:self.ndim]
        operands = [self._data]
        result = list(subscripts)
        basis_out = list(self.basis)
        for i, (bas_curr, bas_out, var, sub) in enumerate(zip(self.basis, basis, self.variance, subscripts)):
            if bas_out == bas_curr:
                continue
            basis_out[i] = bas_out
            # Remove or add basis:
            if is_nobasis(bas_curr) or is_nobasis(bas_out):
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
        subscripts += '->%s' % (''.join(result))
        value = np.einsum(subscripts, *operands, optimize=True)
        return type(self)(value, basis=basis_out)

    # --- Change of basis

    class ChangeBasisInterface:

        def __init__(self, tensor: Tensor) -> None:
            self.tensor = tensor

        def __repr__(self) -> str:
            return f"{type(self).__name__}({self.tensor})"

        def __getitem__(self, key: TBasis) -> Tensor:
            return self.tensor.change_basis(key)

    def change_basis(self, basis: TBasis) -> Tensor:
        basis = BasisTuple.create_from_default(basis, default=self.basis)
        if not basis.is_spanning(self._basis):
            raise BasisError(f"{basis} does not span {self.basis}")
        return self.project(basis)

    @property
    def cob(self):
        return self.ChangeBasisInterface(self)

    def change_basis_at(self, index: int, basis: BasisInterface) -> Tensor:
        if index < 0:
            index += self.ndim
        basis_new = self.basis[:index] + (basis,) + self.basis[index+1:]
        return self.change_basis[basis_new]

    def __or__(self, basis: TBasis) -> Tensor:
        """To allow basis transformation as (array | basis)"""
        # Left-pad with slice(None), such that the basis transformation applies to the last n axes
        basis = BasisTuple.create_from_default(basis, default=self.basis, leftpad=True)
        return self.change_basis(basis)

    def __ror__(self, basis: TBasis) -> Tensor:
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

    def _operator(self, operator, *other, swap=False) -> Tensor:
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
        if isinstance(other, numbers.Number):
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
                 basis: TBasis | None = None,
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

    def transpose(self, axes: tuple[int] | None = None) -> Tensor:
        value = self._data.transpose(axes)
        if axes is None:
            basis = self.basis[::-1]
        else:
            basis = tuple(self.basis[ax] for ax in axes)
        return type(self)(value, basis=basis)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def trace(self, axis1: int = 0, axis2: int = 1) -> Tensor:
        return numpy_functions.trace(self, axis1=axis1, axis2=axis2)

    def dot(self, other: Tensor | np.ndarray) -> Tensor:
        return numpy_functions.dot(self, other)

    # --- Advanced

    def get_hosvd(self, svtol: float | None = None):
        """Calculate core tensor of higher order SVD (HOSVD)."""
        if self.ndim < 3:
            raise NotImplementedError(f"cannot perform HOSVD for {self.ndim}-dimensional tensor.")
        array = self.to_numpy(copy=False)
        core = array
        basis = []
        for dim, bas in enumerate(self.basis):
            if not isinstance(bas, Basis):
                raise RuntimeError
            a = np.moveaxis(array, dim, 0).reshape((array.shape[dim], -1))
            u, s, _ = scipy.linalg.svd(a, check_finite=False)
            if svtol is not None:
                u = u[:, s >= svtol]
            basis.append(bas.make_basis(u))
            core = np.tensordot(core, u.T.conj(), axes=(0, 1))
        return type(self)(core, basis=tuple(basis))


class Cotensor(Tensor):

    DEFAULT_VARIANCE = 1
