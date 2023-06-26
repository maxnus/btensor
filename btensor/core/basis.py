"""
Class hierarchy:

 BasisInterface
    ____|____
   |         |
NoBasis  BasisType
        _____|_____
       |           |
     Basis      Dualbasis
"""


from __future__ import annotations
import functools
from functools import lru_cache
from typing import Optional, Any, Self, TypeAlias, Sequence

import numpy as np

from btensor.util import *
from .space import Space


def is_nobasis(obj):
    return obj is nobasis


def is_basis(obj, allow_nobasis=True, allow_cobasis=True):
    nb = is_nobasis(obj) if allow_nobasis else False
    btype = BasisType if allow_cobasis else Basis
    return isinstance(obj, btype) or nb


class BasisInterface:

    @property
    def id(self) -> int:
        raise NotImplementedError

    def __eq__(self, other):
        """Compare if to bases are the same based on their ID."""
        return isinstance(other, BasisInterface) and hash(self) == hash(other)

    #def __hash__(self) -> int:
    #    raise NotImplementedError

    def __hash__(self) -> int:
        return self.id

    @property
    def variance(self) -> int:
        raise NotImplementedError


TBasis: TypeAlias = BasisInterface | Sequence[BasisInterface]


def compatible_basis(basis1: BasisInterface, basis2: BasisInterface):
    if not (isinstance(basis1, BasisInterface) and isinstance(basis2, BasisInterface)):
        raise TypeError(f"{BasisInterface} required")
    if is_nobasis(basis1) or is_nobasis(basis2):
        return True
    if basis1.variance * basis2.variance == -1:
        return False
    return basis1.is_compatible_with(basis2)


def find_common_parent(basis1: BasisInterface, basis2: BasisInterface) -> BasisInterface:
    if basis1.is_cobasis() ^ basis2.is_cobasis():
        raise ValueError()
    if is_nobasis(basis2):
        return basis1
    if is_nobasis(basis1):
        return basis2
    return basis1.find_common_parent(basis2)


class NoBasis(BasisInterface):

    def __repr__(self):
        return type(self).__name__

    @property
    def id(self) -> int:
        return 0

    #def __hash__(self) -> int:
    #    return 0

    @property
    def variance(self) -> int:
        return 0


nobasis = NoBasis()


class BasisType(BasisInterface):

    def __repr__(self) -> str:
        return f'{type(self).__name__}(id= {self.id}, size= {self.size}, name= {self.name})'

    def __str__(self) -> str:
        if self.name:
            return self.name
        return f'{type(self).__name__}(id= {self.id}, size= {self.size})'

    @property
    def size(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return self.size

    @property
    def root(self):
        raise NotImplementedError

    @property
    def variance(self) -> int:
        raise NotImplementedError

    def dual(self):
        raise NotImplementedError

    def get_nondual(self):
        raise NotImplementedError

    def __invert__(self):
        return self.dual()

    def _get_overlap_mpl(self, other, simplify: bool = False) -> MatrixProductList:
        raise NotImplementedError

    cache_size = 100

    @lru_cache(cache_size)
    def get_overlap(self, other: Self) -> Tensor:
        """Get overlap matrix as an Array with another basis."""
        values = self._get_overlap_mpl(other).evaluate()
        basis = (self.dual(), other.dual())
        return Tensor(values, basis=basis)

    def get_transformation_to(self, other: Self) -> Tensor:
        return self.get_overlap(other.dual())

    def __or__(self, other: Self) -> Self | NotImplemented:
        """Allows writing overlap as `(basis1 | basis2)`."""
        # other might still implement __ror__, so return NotImplemented instead of raising an exception
        if not isinstance(other, BasisType):
            return NotImplemented
        return self.get_overlap(other)

    def __rshift__(self, other: Self) -> Self | NotImplemented:
        if not isinstance(other, BasisType):
            return NotImplemented
        return self.get_transformation_to(other)

    def __lshift__(self, other: Self) -> Self | NotImplemented:
        if not isinstance(other, BasisType):
            return NotImplemented
        return other.get_transformation_to(self)

    def same_root(self, other: Self) -> bool:
        root1 = self.root or self.get_nondual()
        root2 = other.root or other.get_nondual()
        return root1 == root2

    def check_same_root(self, other: Self) -> None:
        if self.same_root(other):
            return
        raise BasisError(f"Bases {self} and {other} do not derive from the same root basis.")


class Basis(BasisType):
    """Basis class.

    Parameters
    ----------
    argument: Array, List, Int
    """

    __next_id = 1

    def __init__(self,
                 argument: int | Sequence[int] | Sequence[bool] | np.ndarray,
                 parent: Optional[Basis] = None,
                 metric: Optional[np.darray] = None,
                 name: Optional[str] = None,
                 debug: bool = False) -> None:
        super().__init__()
        self.parent = parent
        self._id = self._get_next_id()
        self.name = name
        self.debug = debug or getattr(parent, 'debug', False)

        # Root basis:
        if is_int(argument):
            argument = IdentityMatrix(argument)
        # Permutation + selection
        if isinstance(argument, (tuple, list, slice)) or (array_like(argument) and argument.ndim == 1):
            # Convert boolean iterable to indices:
            if ((isinstance(argument, (tuple, list)) or array_like(argument)) and
                    any([isinstance(x, (bool, np.bool_)) for x in argument])):
                argument = np.arange(len(argument))[argument]
            argument = ColumnPermutationMatrix(self.parent.size, argument)
        elif array_like(argument) and argument.ndim == 2:
            argument = GeneralMatrix(argument)
        elif isinstance(argument, Matrix):
            pass
        else:
            raise ValueError("Invalid rotation: %r of type %r" % (argument, type(argument)))
        assert isinstance(argument, Matrix)

        if not self.is_root() and (argument.shape[0] != self.parent.size):
            raise ValueError("Invalid size: %d (expected %d)" % (argument.shape[0], self.parent.size))
        self._coeff = argument
        if self.size == 0:
            raise ValueError("Cannot construct empty basis")

        # Initialize metric matrix:
        if self.is_root():
            if metric is None:
                metric = IdentityMatrix(self.size)
            else:
                metric = SymmetricMatrix(metric)
        else:
            metric_calc = MatrixProductList((argument.T, self.parent.metric, argument)).evaluate()
            if metric is None:
                # Automatically Promote to identity
                idt = IdentityMatrix(self.size)
                if abs(metric_calc - idt.to_array()).max() < 1e-13:
                    metric_calc = idt
                else:
                    metric_calc = SymmetricMatrix(metric_calc)
                metric = metric_calc
            else:
                diff = abs(to_array(metric) - to_array(metric_calc))
                if diff > 1e-8:
                    raise ValueError(f"Large difference between provided and calculated metric matrix "
                                     "(difference= {diff:.1e})")
        self._metric = metric
        if self.debug or __debug__:
            cond = np.linalg.cond(self.metric.to_array())
            if cond > 1e14:
                raise ValueError("Large condition number of metric matrix: %e" % cond)

        # Dual basis
        if self.is_orthonormal:
            self._dual = self
        else:
            self._dual = Dualbasis(self)

    def __class_getitem__(cls, item):
        return functools.partial(cls, parent=item)

    # #@functools.cache
    # def __hash__(self) -> int:
    #     print(type(self.coeff))
    #     return int(hashlib.sha256(self.coeff).hexdigest(), 16)

    @property
    def id(self) -> int:
        return self._id

    @property
    def size(self) -> int:
        """Number of basis functions."""
        return self.coeff.shape[1]

    @property
    def variance(self) -> int:
        return 1

    @property
    def coeff(self) -> np.ndarray:
        return self._coeff

    @property
    def metric(self) -> Optional[Matrix]:
        return self._metric

    @property
    def root(self) -> Optional[Self]:
        if self.parent is None:
            return None
        if self.parent.root is None:
            return self.parent
        return self.parent.root

    def is_root(self) -> bool:
        return self.parent is None

    @property
    def space(self) -> Space:
        return Space(self)

    def coeff_in_basis(self, basis) -> MatrixProductList:
        """Express coeffients in different (parent) basis (rather than the direct parent).

        Was BUGGY before, now fixed?"""
        if not is_basis(basis):
            raise TypeError

        if basis == self:
            return MatrixProductList([IdentityMatrix(self.size)])

        self.check_same_root(basis)
        parents = self.get_parents()
        nondual = basis.get_nondual()
        if nondual not in parents:
            raise ValueError(f"{basis} is not superbasis of {self}")
        matrices = []
        for p in parents:
            if p == nondual:
                break
            matrices.append(p.coeff)

        if basis.is_cobasis():
            raise NotImplementedError
            #matrices.append(basis.metric)

        matrices = matrices[::-1]
        matrices.append(self.coeff)
        return MatrixProductList(matrices)

    @staticmethod
    def _get_next_id() -> int:
        next_id = Basis.__next_id
        Basis.__next_id += 1
        return next_id

    def make_basis(self, *args, **kwargs) -> Basis:
        """Make a new basis with coefficients or indices in reference to the current basis."""
        basis = Basis(*args, parent=self, **kwargs)
        return basis

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

    def is_compatible_with(self, other: BasisInterface) -> bool:
        return is_nobasis(other) or self.same_root(other)

    def find_common_parent(self, other: BasisType) -> Basis:
        """Find first common ancestor between two bases."""
        if other.is_cobasis():
            raise ValueError
        self.check_same_root(other)
        parents1 = self.get_parents(include_self=True)[::-1]
        parents2 = other.get_parents(include_self=True)[::-1]
        assert (parents1[0] is parents2[0])
        parent = None
        for i, p in enumerate(parents1):
            if i >= len(parents2) or p != parents2[i]:
                break
            parent = p
        assert parent is not None
        return parent

    def _get_overlap_mpl(self, other, simplify=False) -> MatrixProductList:
        """Return MatrixProduct required for as_basis method"""
        self.check_same_root(other)
        # Find first common ancestor and express coefficients in corresponding basis
        parent = self.find_common_parent(other.get_nondual())
        matprod = self.coeff_in_basis(parent).T + [parent.metric] + other.coeff_in_basis(parent)
        if simplify:
            matprod = matprod.simplify()
        return matprod

    def dual(self) -> Dualbasis | Self:
        return self._dual

    def get_nondual(self) -> Self:
        return self

    @staticmethod
    def is_cobasis() -> bool:
        return False

    @property
    def is_orthonormal(self) -> bool:
        return isinstance(self.metric, IdentityMatrix)

    def is_derived_from(self, other: Basis, inclusive: bool = False) -> bool:
        """True if self is derived from other, else False"""
        if not self.same_root(other):
            return False
        for parent in self.get_parents(include_self=inclusive):
            if other == parent:
                return True
        return False

    def is_parent_of(self, other: Basis, inclusive: bool = False) -> bool:
        """True if self is parent of other, else False"""
        return other.is_derived_from(self, inclusive=inclusive)

    def get_orthonormal_error(self) -> float:
        ortherr = abs(self.metric-np.identity(self.size)).max()
        return ortherr

    def __neg__(self) -> Dualbasis | Self:
        return self.dual()


class Dualbasis(BasisType):

    def __init__(self, basis: Basis) -> None:
        super().__init__()
        self._basis = basis

    @property
    def id(self) -> int:
        return -(self.dual().id)

    @property
    def size(self) -> int:
        return self.dual().size

    @property
    def variance(self) -> int:
        return -1

    def dual(self) -> Basis:
        return self._basis

    def get_nondual(self) -> Basis:
        return self.dual()

    @property
    def name(self) -> str:
        return f"{type(self).__name__}{self.dual().name})"

    @staticmethod
    def is_dual() -> bool:
        return True

    @property
    def root(self) -> Optional[Basis]:
        return self.dual().root

    def coeff_in_basis(self, basis):
        matrices = self.get_nondual().coeff_in_basis(basis)
        # To raise right-hand index:
        matrices.append(self.metric)
        return matrices

    @property
    def metric(self):
        return self.dual().metric.inverse

    def _get_overlap_mpl(self, other, simplify=False):
        """Append inverse metric (metric of dual space)"""
        matprod = [self.metric] + self.get_nondual()._get_overlap_mpl(other)
        if simplify:
            matprod = matprod.simplify()
        return matprod

    def __pos__(self) -> Basis:
        return self.dual()

    @staticmethod
    def is_cobasis() -> bool:
        return True


from .tensor import Tensor
