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
from functools import lru_cache
from typing import *

from loguru import logger
import numpy as np
import scipy

from btensor.util import (Matrix, IdentityMatrix, SymmetricMatrix, ColumnPermutationMatrix, GeneralMatrix,
                          MatrixProductList, is_int, array_like, BasisError)
from btensor.space import Space


def _is_nobasis(obj):
    return obj is nobasis


def _is_basis(obj, allow_nobasis=True):
    nb = _is_nobasis(obj) if allow_nobasis else False
    return isinstance(obj, Basis) or nb


class Variance:
    INVARIANT = 0
    COVARIANT = 1
    CONTRAVARIANT = -1


class BasisInterface:

    @property
    def id(self) -> int:
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__

    def __eq__(self, other):
        """Compare if to bases are the same based on their ID."""
        return isinstance(other, BasisInterface) and hash(self) == hash(other)

    def __hash__(self) -> int:
        return self.id


BasisT: TypeAlias = Union[BasisInterface, Sequence[BasisInterface]]
BasisDefinitionT: TypeAlias = Union[int, Sequence[int], Sequence[bool], slice, np.ndarray]


def compatible_basis(basis1: BasisInterface, basis2: BasisInterface):
    if not (isinstance(basis1, BasisInterface) and isinstance(basis2, BasisInterface)):
        raise TypeError(f"{BasisInterface} required")
    if _is_nobasis(basis1) or _is_nobasis(basis2):
        return True
    return basis1.is_compatible_with(basis2)


def get_common_parent(basis1: BasisInterface, basis2: BasisInterface) -> BasisInterface:
    if _is_nobasis(basis2):
        return basis1
    if _is_nobasis(basis1):
        return basis2
    return basis1.get_common_parent(basis2)


class NoBasis(BasisInterface):

    @property
    def id(self) -> int:
        return 0


nobasis = NoBasis()


class Basis(BasisInterface):
    """Basis class.

    Parameters
    ----------
    definition: Array, List, Int
    """
    __next_id = 1

    def __init__(self,
                 definition: BasisDefinitionT,
                 parent: Basis | None = None,
                 metric: np.darray | None = None,
                 name: str | None = None,
                 orthonormal: bool = False) -> None:
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
        self.name = name
        self._matrix = self.definition_to_matrix(definition)
        if metric is None:
            if orthonormal or self.is_root():
                metric = IdentityMatrix(self.size)
            else:
                metric = SymmetricMatrix(MatrixProductList([self.matrix.T, self.parent.metric, self.matrix]).evaluate())
        elif orthonormal:
            raise ValueError(f"orthonormal basis cannot have a metric")
        elif isinstance(metric, np.ndarray):
            metric = SymmetricMatrix(metric)
        self._metric = metric
        self._intersect_cache = {}

    def definition_to_matrix(self, definition: BasisDefinitionT) -> Matrix:
        # Root basis:
        if is_int(definition):
            matrix = IdentityMatrix(definition)
        # Permutation + selection
        elif isinstance(definition, (tuple, list, slice)) or (array_like(definition) and definition.ndim == 1):
            # Convert boolean iterable to indices:
            if ((isinstance(definition, (tuple, list)) or array_like(definition)) and
                    any([isinstance(x, (bool, np.bool_)) for x in definition])):
                definition = np.arange(len(definition))[definition]
            matrix = ColumnPermutationMatrix(self.parent.size, definition)
        elif array_like(definition) and definition.ndim == 2:
            matrix = GeneralMatrix(definition)
        elif isinstance(definition, Matrix):
            matrix = definition
        else:
            raise ValueError(f"invalid basis definition: {definition} of type {type(definition)}")
        if not self.is_root() and (matrix.shape[0] != self.parent.size):
            raise ValueError(f"invalid basis definition size: {matrix.shape[0]} (expected {self.parent.size})")
        return matrix

    # --- Basis properties and methods

    @property
    def id(self) -> int:
        return self._id

    @staticmethod
    def _get_next_id() -> int:
        next_id = Basis.__next_id
        Basis.__next_id += 1
        return next_id

    def __repr__(self) -> str:
        return f'{type(self).__name__}(id= {self.id}, size= {self.size}, name= {self.name})'

    @property
    def size(self) -> int:
        """Number of basis functions."""
        return self.matrix.shape[1]

    def __len__(self) -> int:
        return self.size

    @property
    def parent(self):
        return self._parent

    @property
    def root(self) -> Basis | None:
        return self._root

    @property
    def matrix(self) -> Matrix:
        return self._matrix

    @property
    def metric(self) -> Matrix:
        return self._metric

    @property
    def space(self) -> Space:
        return Space(self)

    @property
    def is_orthonormal(self) -> bool:
        return isinstance(self.metric, IdentityMatrix)

    # --- Make new basis

    def make_subbasis(self, *args, name: str | None = None, orthonormal: bool = False, **kwargs) -> Basis:
        """Make a new basis with coefficients or indices in reference to the current basis."""
        return type(self)(*args, parent=self, name=name, orthonormal=orthonormal, **kwargs)

    def make_union_basis(self, *other: Basis, tol: float = 1e-12, name: str | None = None) -> Basis:
        """Make smallest possible orthonormal basis, which spans both self and other."""
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
        return common_parent.make_subbasis(v, name=name, orthonormal=common_parent.is_orthonormal)

    def make_intersect_basis(self,
                             *other: Basis,
                             tol: float = 1e-12,
                             name: str | None = None,
                             cache: bool = True) -> Basis:
        if len(other) > 1:
            # TODO
            raise NotImplementedError
        other = other[0]

        # Caching
        cache_key = (other.id, tol)
        if (cached := self._intersect_cache.get(cache_key, None)) is not None:
            logger.debug("returning cached result")
            return cached

        # Alternative method (works only for orthonormal basis?)
        #m = parent.get_transformation_to(non_parent).to_numpy()
        #u, s, vh = scipy.linalg.svd(m, full_matrices=False)
        #u = u[:, s >= 1-tol]
        #return parent.make_basis(u, name=name, orthonormal=parent.is_orthonormal)

        s = self.get_overlap(other).to_numpy()
        if other.is_orthonormal:
            p = np.dot(s, s.T)
        else:
            p = np.linalg.multi_dot([s, other.metric.inverse.to_numpy(), s.T])
        e, v = scipy.linalg.eigh(p, b=self.metric.to_numpy())
        logger.debug("tolerance= {}, eigenvalues= {}", tol, e)
        v = v[:, e >= tol]
        intersect = self.make_subbasis(v, name=name, orthonormal=self.is_orthonormal)
        logger.debug("basis sizes: self= {}, other= {}, intersect= {}", len(self), len(other), len(intersect))
        if cache:
            logger.debug("storing result in cache")
            self._intersect_cache[cache_key] = intersect
        return intersect

    def _projector_in_basis(self, basis: Basis) -> np.ndarray:
        """Projector onto self in basis."""
        c = self.coeff_in_basis(basis)
        p = c + c.T if self.is_orthonormal else c + [self.metric] + c.T
        return p.evaluate()

    def is_root(self) -> bool:
        return self.parent is None

    def get_parents(self, include_root: bool = True, include_self: bool = False) -> list[Basis]:
        """Get list of parent bases ordered from direct parent to root basis."""
        parents = [self] if include_self else []
        current = self
        while current.parent is not None:
            parents.append(current.parent)
            current = current.parent
        if not include_root:
            parents = parents[:-1]
        return parents

    def is_derived_from(self, other: Basis, inclusive: bool = False) -> bool:
        """True if self is derived from other, else False"""
        if not self.same_root(other):
            return False
        return other in self.get_parents(include_self=inclusive)

    def is_parent_of(self, other: Basis, inclusive: bool = False) -> bool:
        """True if self is parent of other, else False"""
        return other.is_derived_from(self, inclusive=inclusive)

    # --- Methods taking another basis instance

    def is_compatible_with(self, other: BasisInterface) -> bool:
        return _is_nobasis(other) or self.same_root(other)

    def same_root(self, *other: Basis) -> bool:
        roots = np.asarray([basis.root or basis for basis in [self, *other]])
        return np.all(roots == roots[0])

    def check_same_root(self, *other: Basis) -> None:
        if self.same_root(*other):
            return
        raise BasisError(f"Bases {self} and {other} do not derive from the same root basis.")

    def get_common_parent(self, *other: Basis) -> Basis:
        """Find first common ancestor between multiple bases."""
        # support multiple bases via recursion
        if len(other) > 1:
            parent = self.get_common_parent(other[0])
            return parent.get_common_parent(*other[1:])
            #common_parent = other[-2].get_common_parent(other[-1])
            #return self.get_common_parent(*(other[:-2] + (common_parent,)))
        if len(other) != 1:
            raise ValueError
        other = other[0]
        self.check_same_root(other)
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

    def coeff_in_basis(self, basis: Basis) -> MatrixProductList:
        """Express coeffients in different (parent) basis (rather than the direct parent)."""
        if basis == self:
            return MatrixProductList([IdentityMatrix(self.size)])

        self.check_same_root(basis)
        parents = self.get_parents()
        if basis not in parents:
            raise ValueError(f"{basis} is not superbasis of {self}")
        matrices = []
        for p in parents:
            if p == basis:
                break
            matrices.append(p.matrix)
        matrices = matrices[::-1]
        matrices.append(self.matrix)
        return MatrixProductList(matrices)

    def _get_overlap_mpl(self, other: Basis, variance: tuple[int, int] | None = None) -> MatrixProductList:
        """Return MatrixProduct required for as_basis method"""
        self.check_same_root(other)
        if variance is None:
            variance = (Variance.COVARIANT, Variance.COVARIANT)
        # Find first common ancestor and express coefficients in corresponding basis
        parent = self.get_common_parent(other)
        if variance[0] == Variance.CONTRAVARIANT:
            mpl = [self.metric.inverse]
        else:
            mpl = []
        mpl = MatrixProductList(mpl)
        mpl += self.coeff_in_basis(parent).T + [parent.metric] + other.coeff_in_basis(parent)
        if variance[1] == Variance.CONTRAVARIANT:
            mpl += [other.metric.inverse]
        return mpl

    cache_size = 100

    @lru_cache(cache_size)
    def get_overlap(self, other: Basis, variance: tuple[int, int] = (Variance.COVARIANT, Variance.COVARIANT)) -> Tensor:
        """Get overlap matrix as an Array with another basis."""
        values = self._get_overlap_mpl(other, variance=variance).evaluate()
        return Tensor(values, basis=(self, other), variance=variance)

    def get_transformation_to(self, other: Basis) -> Tensor:
        return self.get_overlap(other, variance=(Variance.CONTRAVARIANT, Variance.COVARIANT))

    def get_transformation_from(self, other: Basis) -> Tensor:
        return other.get_overlap(self, variance=(Variance.CONTRAVARIANT, Variance.COVARIANT))


from btensor.tensor import Tensor
