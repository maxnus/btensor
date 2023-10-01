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

"""Module defining the Basis class and related functions."""

from __future__ import annotations
from functools import lru_cache
import weakref
from typing import *

import numpy as np
import scipy

from btensor.util import (Matrix, IdentityMatrix, SymmetricMatrix, ColumnPermutationMatrix, GeneralMatrix,
                          MatrixProductList, is_int, array_like)
from btensor.exceptions import BasisError
from btensor.space import Space


def _is_nobasis(obj):
    return obj is nobasis


def _is_basis_or_nobasis(obj):
    return isinstance(obj, Basis) or obj is nobasis


class _Variance:
    INVARIANT = 0
    COVARIANT = 1
    CONTRAVARIANT = -1


class _NoBasis:
    """Class for nobasis singleton."""

    def __repr__(self):
        return type(self).__name__.lower()


nobasis = _NoBasis()

IBasis: TypeAlias = Union['Basis', _NoBasis]
NBasis: TypeAlias = Union[IBasis, Sequence[IBasis]]
BasisArgument: TypeAlias = Union[Sequence[int], Sequence[bool], slice, np.ndarray]


def compatible_basis(basis1: IBasis, basis2: IBasis):
    """Check if two bases are compatible with each other.

    Compatible means that the two Tensors with this basis can be added, subtracted, contracted, etc.

    Args:
        basis1, basis2: The two bases to be checked.

    Returns:
        True if both bases are compatible, False otherwise.
    """
    if not (_is_basis_or_nobasis(basis1) and _is_basis_or_nobasis(basis2)):
        raise TypeError(f"{Basis} or {nobasis} required")
    if _is_nobasis(basis1) or _is_nobasis(basis2):
        return True
    return basis1.is_compatible_with(basis2)


def get_common_parent(basis1: IBasis, basis2: IBasis) -> IBasis:
    """Get common parent of two bases.

    Args:
        basis1: Basis 1.
        basis2: Basis 2.

    Returns:
        Common parent of `basis1` and `basis2`. Returns `nobasis` if both bases are equal `nobasis`.
    """
    if _is_nobasis(basis2):
        return basis1
    if _is_nobasis(basis1):
        return basis2
    return basis1.get_common_parent(basis2)


class Basis:
    """Class to represent a vector space basis, which can be used to define a Tensor object.

    Args:
        argument: Integer to create a rootbasis or sequence, slice, or array to create a derived basis.
        parent: Parent basis object for the derived basis. If it is None, the basis is a root-basis. Default: None.
        metric: Metric array, representing the inner product of the basis with itself. It is used to raise and lower
            tensor indices, if the basis is non-orthonormal. Default: None.
        name: Name of the basis, which is used for string representations of basis objects.
            Basis names are not required to be unique. Default: None.
        orthonormal: Set True, if the basis is orthonormal, i.e. the metric is the identity matrix. If None,
            the basis is assumed orthonormal if a) it is a root basis without metric or b) it is a derived basis
            with an orthonormal parent basis, defined in terms of a permutation + selection (slice or 1D sequence).
            Default: None.
    """
    # Private (inheriting classes will have their own version)
    # ID used for next created Basis object:
    __next_id = 1
    # Keep a weak reference of all created bases:
    __basis_by_id = weakref.WeakValueDictionary()

    def __init__(self,
                 argument: int | BasisArgument,
                 *,
                 parent: Basis | None = None,
                 metric: np.ndarray | None = None,
                 name: str | None = None,
                 orthonormal: bool | None = None) -> None:
        """Initialize new Basis object."""
        super().__init__()
        self._parent = parent
        if parent is None:
            self._root = None
        elif parent.root is None:
            self._root = parent
        else:
            self._root = parent.root
        self._id = self._get_next_id()
        if name is None:
            name = f'Basis{self._id}'
        self._name = name
        self._matrix = self._argument_to_matrix(argument)
        if orthonormal is None:
            orthonormal = (self.is_root() and metric is None) or (not self.is_root() and self.parent.is_orthonormal
                                                                  and isinstance(self._matrix, ColumnPermutationMatrix))
        if metric is None:
            if orthonormal:
                metric = IdentityMatrix(self.size)
            else:
                metric = SymmetricMatrix(MatrixProductList([self._matrix.T, self.parent.metric, self._matrix]).evaluate())
        elif orthonormal:
            raise ValueError(f"orthonormal basis cannot have a metric")
        elif isinstance(metric, np.ndarray):
            metric = SymmetricMatrix(metric)
        self._metric = metric
        self._space = Space(self)
        self._union_cache: Dict[Tuple[float | int, ...], int] = {}
        self._intersect_cache: Dict[Tuple[float | int, ...], int] = {}
        self.__basis_by_id[self.id] = self

    T = TypeVar('T')

    @classmethod
    def get_by_id(cls, id: int, default: T = None) -> Basis | T:
        """Returns a previously defined basis based on its ID, if it exists.

        Note that the basis may not be found because it has been garbage collected.

        Args:
            id: The ID of the basis.
            default: If the basis does not exist, default will be returned instead.

        Returns:
            Existing basis or default.
        """
        return cls.__basis_by_id.get(id, default)

    def _argument_to_matrix(self, argument: int | BasisArgument) -> Matrix:
        # Root basis:
        if is_int(argument):
            matrix = IdentityMatrix(argument)
        # Permutation + selection
        elif isinstance(argument, (tuple, list, slice)) or (array_like(argument) and argument.ndim == 1):
            # Convert boolean iterable to indices:
            if ((isinstance(argument, (tuple, list)) or array_like(argument)) and
                    any([isinstance(x, (bool, np.bool_)) for x in argument])):
                argument = np.arange(self.parent.size)[argument]
            matrix = ColumnPermutationMatrix(self.parent.size, argument)
        elif array_like(argument) and argument.ndim == 2:
            matrix = GeneralMatrix(argument)
        elif isinstance(argument, Matrix):
            matrix = argument
        else:
            raise ValueError(f"invalid basis argument: {argument} of type {type(argument)}")
        if not self.is_root() and (matrix.shape[0] != self.parent.size):
            raise ValueError(f"invalid basis argument size: {matrix.shape[0]} (expected {self.parent.size})")
        return matrix

    # --- Basis properties and methods

    @property
    def id(self) -> int:
        """Unique ID of basis."""
        return self._id

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other):
        """Compare if to bases are the same based on their ID."""
        return isinstance(other, Basis) and hash(self) == hash(other)

    @staticmethod
    def _get_next_id() -> int:
        next_id = Basis.__next_id
        Basis.__next_id += 1
        return next_id

    def __repr__(self) -> str:
        return f'{type(self).__name__}(id= {self.id}, size= {self.size}, name= {self.name})'

    @property
    def name(self) -> str:
        """Name of basis."""
        return self._name

    @property
    def size(self) -> int:
        """Size of basis."""
        return self._matrix.shape[1]

    def __len__(self) -> int:
        return self.size

    @property
    def parent(self) -> Basis | None:
        """Parent of basis or None, if the basis is a root-basis."""
        return self._parent

    @property
    def root(self) -> Basis | None:
        """Root-basis of the basis or None, if the basis itself is the root-basis."""
        return self._root

    @property
    def metric(self) -> Matrix:
        """Metric matrix, used to raise and lower tensor indices."""
        return self._metric

    @property
    def space(self) -> Space:
        """Space spanned by basis."""
        return self._space

    @property
    def is_orthonormal(self) -> bool:
        """True if the basis is orthonormal."""
        return isinstance(self.metric, IdentityMatrix)

    # --- Make new basis

    def make_subbasis(self,
                      argument: BasisArgument,
                      *,
                      metric: np.ndarray | None = None,
                      name: str | None = None,
                      orthonormal: bool = False) -> Basis:
        """Make a new basis with coefficients or indices in reference to the current basis.

        Args:
            argument: Sequence, slice, or array defining the sub-basis with respect to the parent basis.
            metric: Metric of the sub-basis. Default: None.
            name: Name of sub-basis. Default: None.
            orthonormal: True if the sub-basis is orthonormal. Default: False.

        Returns:
            Sub-basis of parent.
        """
        return type(self)(argument, parent=self, metric=metric, name=name, orthonormal=orthonormal)

    def make_union_basis(self,
                         *other: Basis,
                         tol: float = 1e-12,
                         name: str | None = None,
                         cache: bool = True) -> Basis:
        """Make the smallest orthonormal basis, which spans both the basis and one or more other bases.

        Args:
            *other: One or more other bases.
            tol: Tolerance used to construct the union basis via the eigendecomposition of a projection matrix.
                Default: 1e-12.
            name: Name of union basis. Default: None.
            cache: Cache resulting intersection basis, for future use. Default: True.

        Returns:
            Union basis, spanning joined space of all input bases.
        """
        # Caching
        cache_key = (tol, *(x.id for x in other))
        try:
            if (cached := self.get_by_id(self._union_cache[cache_key])) is not None:
                return cached
        except KeyError:
            pass

        common_parent = self.get_common_parent(*other)
        if common_parent in [self, *other]:
            return common_parent
        m = self._projector_in_basis(common_parent)
        for other_basis in other:
            m += other_basis._projector_in_basis(common_parent)
        #metric = common_parent.metric.to_numpy() if not common_parent.is_orthonormal else None
        #e, v = scipy.linalg.eigh(m, b=metric)
        # metric should not be here?
        e, v = np.linalg.eigh(m)
        v = v[:, e >= tol]

        union = common_parent.make_subbasis(v, name=name, orthonormal=common_parent.is_orthonormal)
        if cache:
            self._union_cache[cache_key] = union.id
        return union

    def make_intersect_basis(self,
                             *other: Basis,
                             tol: float = 1e-12,
                             name: str | None = None,
                             use_svd: bool = True,
                             cache: bool = True) -> Basis:
        """Make the smallest orthonormal basis, which spans the intersecting space of both the basis and another basis.

        Args:
            *other: One other basis.
            tol: Tolerance used to construct the intersection basis via the eigendecomposition of a projection matrix.
                Default: 1e-12.
            use_svd: Perform singular-value decomposition (SVD) instead of eigendecompostion, if possible.
                Default: True.
            name: Name of intersection basis. Default: None.
            cache: Cache resulting intersection basis, for future use. Default: True.

        Returns:
            Intersection basis, spanning intersecting space of all input bases.
        """
        # Caching
        cache_key = (tol, *(x.id for x in other))
        try:
            if (cached := self.get_by_id(self._intersect_cache[cache_key])) is not None:
                return cached
        except KeyError:
            pass

        if len(other) > 1:
            # TODO
            raise NotImplementedError
        other = other[0]

        # Via SVD (works only if bases are orthonormal? It might only require one basis to be orthonormal):
        if use_svd and (self.is_orthonormal and other.is_orthonormal):
            m = self.get_transformation_to(other).to_numpy()
            vl, s, vr = scipy.linalg.svd(m, full_matrices=False)
            v = vl[:, s**2 >= tol]
        # Via eigendecomposition:
        else:
            s = self.get_overlap(other).to_numpy()
            if other.is_orthonormal:
                p = np.dot(s, s.T)
            else:
                p = np.linalg.multi_dot([s, other.metric.inverse.to_numpy(), s.T])
            b = self.metric.to_numpy() if not self.is_orthonormal else None
            e, v = scipy.linalg.eigh(p, b=b)
            v = v[:, e >= tol]

        intersect = self.make_subbasis(v, name=name, orthonormal=self.is_orthonormal)
        if cache:
            self._intersect_cache[cache_key] = intersect.id
        return intersect

    def _projector_in_basis(self, basis: Basis) -> np.ndarray:
        """Projector onto self in basis."""
        c = self._coeff_in_basis(basis)
        p = c + c.T if self.is_orthonormal else c + [self.metric] + c.T
        return p.evaluate()

    def is_root(self) -> bool:
        """True if basis is a root-basis, False otherwise."""
        return self.parent is None

    def get_parents(self, *, include_root: bool = True, include_self: bool = False) -> list[Basis]:
        """Get list of parent bases ordered from direct parent to root basis.

        Args:
            include_root: Include the root-basis in the list of parents. Default: True.
            include_self: Include the current basis itself in the list of parents: Default: False.

        Returns:
            List of parent bases.
        """
        parents = [self] if include_self else []
        current = self
        while current.parent is not None:
            parents.append(current.parent)
            current = current.parent
        if not include_root:
            parents = parents[:-1]
        return parents

    def is_derived_from(self, other: Basis, inclusive: bool = False) -> bool:
        """Check if the basis is derived from a second basis.

        Args:
            other: Second basis.
            inclusive: If True, the function will return True, even if the current basis and `other` are the same.
                Default: False.

        Returns:
            True if `other` is derived from the basis, False otherwise.
        """
        if not self.same_root(other):
            return False
        return other in self.get_parents(include_self=inclusive)

    def is_parent_of(self, other: Basis, inclusive: bool = False) -> bool:
        """Check if the basis is parent of a second basis.

        Args:
            other: Second basis.
            inclusive: If True, the function will return True, even if the current basis and `other` are the same.
                Default: False.

        Returns:
            True if `other` is a parent of the basis, False otherwise.
        """
        return other.is_derived_from(self, inclusive=inclusive)

    # --- Methods taking another basis instance

    def is_compatible_with(self, other: IBasis) -> bool:
        """Check if basis is compatible with another basis.

        Compatible means that the two Tensors with this basis can be added, subtracted, contracted, etc.

        Args:
            other: The second basis.

        Returns:
            True if both bases are compatible, False otherwise.
        """
        return _is_nobasis(other) or self.same_root(other)

    def same_root(self, *other: Basis) -> bool:
        """Check if all bases have the same root.

        Args:
            *other: One or multiple other bases.

        Returns:
            True if all bases have the same root-basis, False otherwise.
        """
        if len(other) == 1:
            return (self.root or self) == (other[0].root or other[0])

        roots = np.asarray([basis.root or basis for basis in other])
        return np.all(roots == (self.root or self))

    def _check_same_root(self, *other: Basis) -> None:
        """Check if all bases have the same root and raise a BasisError if not."""
        if self.same_root(*other):
            return
        raise BasisError(f"Bases {self} and {other} do not derive from the same root basis.")

    def get_common_parent(self, *other: Basis) -> Basis:
        """Find first common ancestor basis between multiple bases.

        Args:
            *other: One or multiple other bases.

        Returns:
            Basis which is the first common parent between all bases.
        """
        # support multiple bases via recursion
        if len(other) > 1:
            parent = self.get_common_parent(other[0])
            return parent.get_common_parent(*other[1:])
            #common_parent = other[-2].get_common_parent(other[-1])
            #return self.get_common_parent(*(other[:-2] + (common_parent,)))
        if len(other) != 1:
            raise ValueError
        other = other[0]
        self._check_same_root(other)
        parents_self = self.get_parents(include_self=True)[::-1]
        parents_other = other.get_parents(include_self=True)[::-1]
        assert (parents_self[0] is parents_other[0])
        common_parent = None
        for i, p in enumerate(parents_self):
            if i >= len(parents_other) or p != parents_other[i]:
                break
            common_parent = p
        assert common_parent is not None
        return common_parent

    def _coeff_in_basis(self, basis: Basis) -> MatrixProductList:
        """Express coeffients in different (parent) basis (rather than the direct parent)."""
        if basis == self:
            return MatrixProductList([IdentityMatrix(self.size)])

        self._check_same_root(basis)
        parents = self.get_parents()
        if basis not in parents:
            raise ValueError(f"{basis} is not superbasis of {self}")
        matrices = []
        for p in parents:
            if p == basis:
                break
            matrices.append(p._matrix)
        matrices = matrices[::-1]
        matrices.append(self._matrix)
        return MatrixProductList(matrices)

    def _get_overlap_mpl(self, other: Basis, variance: tuple[int, int] | None = None) -> MatrixProductList:
        """Return MatrixProduct."""
        self._check_same_root(other)
        if variance is None:
            variance = (_Variance.COVARIANT, _Variance.COVARIANT)
        # Find first common ancestor and express coefficients in corresponding basis
        parent = self.get_common_parent(other)
        if variance[0] == _Variance.CONTRAVARIANT:
            mpl = [self.metric.inverse]
        else:
            mpl = []
        mpl = MatrixProductList(mpl)
        mpl += self._coeff_in_basis(parent).T + [parent.metric] + other._coeff_in_basis(parent)
        if variance[1] == _Variance.CONTRAVARIANT:
            mpl += [other.metric.inverse]
        return mpl

    _transformation_cache_size = 128

    @lru_cache(_transformation_cache_size)
    def get_transformation(self,
                           other: Basis,
                           variance: tuple[int, int] = (_Variance.COVARIANT, _Variance.COVARIANT)) -> Tensor:
        """Get transformation matrix to another basis as a Tensor with general variance.

        Args:
            other: Second basis.
            variance: Variance of transformation matrix.

        Returns:
            Transformation matrix.
        """
        values = self._get_overlap_mpl(other, variance=variance).evaluate()
        return Tensor(values, basis=(self, other), variance=variance, copy_data=False)

    def get_overlap(self, other: Basis) -> Tensor:
        """Get overlap matrix with another basis as a Tensor.

        Args:
            other: Second basis.

        Returns:
            Overlap matrix.
        """
        return self.get_transformation(other)

    def get_transformation_to(self, other: Basis) -> Tensor:
        """Get transformation matrix to another basis as a Tensor with variance (-1, 1).

        Args:
            other: Second basis.

        Returns:
            Transformation matrix with variance (-1, 1).
        """
        return self.get_transformation(other, variance=(_Variance.CONTRAVARIANT, _Variance.COVARIANT))

    def get_transformation_from(self, other: Basis) -> Tensor:
        """Get transformation matrix from another basis as a Tensor with variance (-1, 1).

        Args:
            other: Second basis.

        Returns:
            Transformation matrix with variance (-1, 1).
        """
        return other.get_transformation(self, variance=(_Variance.CONTRAVARIANT, _Variance.COVARIANT))


from btensor.tensor import Tensor
