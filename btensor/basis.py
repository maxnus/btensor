import functools
from functools import lru_cache
import hashlib

import numpy as np

from btensor.util import *
from btensor.space import Space


def is_basis(obj, allow_nobasis=True, allow_cobasis=True):
    nb = is_nobasis(obj) if allow_nobasis else False
    btype = BasisOrDualBasis if allow_cobasis else Basis
    return isinstance(obj, btype) or nb


def is_nobasis(obj):
    return obj is nobasis


class BasisType:

    @property
    def id(self) -> int:
        raise NotImplementedError

    def __eq__(self, other):
        """Compare if to bases are the same based on their ID."""
        return isinstance(other, BasisType) and hash(self) == hash(other)

    #def __hash__(self) -> int:
    #    raise NotImplementedError

    def __hash__(self) -> int:
        return self.id

    @property
    def variance(self) -> int:
        raise NotImplementedError


def compatible_basis(basis1: BasisType, basis2: BasisType):
    if not (isinstance(basis1, BasisType) and isinstance(basis2, BasisType)):
        raise TypeError
    if is_nobasis(basis1) or is_nobasis(basis2):
        return True
    if basis1.variance * basis2.variance == -1:
        return False
    return basis1.compatible(basis2)


def find_common_parent(basis1: BasisType, basis2: BasisType) -> BasisType:
    if basis1.is_cobasis() ^ basis2.is_cobasis():
        raise ValueError()
    if is_nobasis(basis2):
        return basis1
    if is_nobasis(basis1):
        return basis2
    return basis1.find_common_parent(basis2)


class NoBasis(BasisType):

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


class BasisOrDualBasis(BasisType):

    def __repr__(self) -> str:
        return f'{type(self).__name__}(id= {self.id}, size= {self.size}, name= {self.name})'

    def __str__(self) -> str:
        if self.name:
            return self.name
        return f'{type(self).__name__}(id= {self.id}, size= {self.size})'

    @property
    def size(self):
        raise NotImplementedError

    def __len__(self):
        return self.size

    @property
    def root(self):
        raise NotImplementedError

    def dual(self):
        raise NotImplementedError

    def get_nondual(self):
        raise NotImplementedError

    def __invert__(self):
        return self.dual()

    def _as_basis_matprod(self, other, simplify=False):
        raise NotImplementedError

    cache_size = 100

    @lru_cache(cache_size)
    def as_basis(self, other):
        """Get overlap matrix as an Array with another basis."""
        matprod = self._as_basis_matprod(other)
        return Tensor(matprod.evaluate(), basis=(~other, ~self))

    def __or__(self, other):
        """Allows writing overlap as `(basis1 | basis2)`."""
        # other might still implement __ror__, so return NotImplemented instead of raising an exception
        if not isinstance(other, BasisOrDualBasis):
            return NotImplemented
        return other.as_basis(self)

    def same_root(self, other):
        root1 = self.root or self.get_nondual()
        root2 = other.root or other.get_nondual()
        return root1 == root2

    def check_same_root(self, other):
        if not self.same_root(other):
            raise BasisError(f"Bases {self} and {other} do not derive from the same root basis.")


class Basis(BasisOrDualBasis):
    """Basis class.

    Parameters
    ----------
    argument: Array, List, Int
    """

    __next_id = 1

    def __init__(self, argument, parent=None, metric=None, name=None, debug=False, **kwargs):
        super().__init__(**kwargs)
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
            metric_calc = MatrixProduct((argument.T, self.parent.metric, argument)).evaluate()
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
            self._dual = Cobasis(self)

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
    def coeff(self):
        return self._coeff

    @property
    def metric(self):
        return self._metric

    @property
    def root(self):
        if self.parent is None:
            return None
        if self.parent.root is None:
            return self.parent
        return self.parent.root

    def is_root(self):
        return self.parent is None

    @property
    def space(self):
        return Space(self)

    def coeff_in_basis(self, basis):
        """Express coeffients in different (parent) basis (rather than the direct parent).

        Was BUGGY before, now fixed?"""
        if not is_basis(basis):
            raise TypeError

        if basis == self:
            return MatrixProduct([IdentityMatrix(self.size)])

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
        return MatrixProduct(matrices)

    @staticmethod
    def _get_next_id():
        next_id = Basis.__next_id
        Basis.__next_id += 1
        return next_id

    def make_basis(self, *args, **kwargs):
        """Make a new basis with coefficients or indices in reference to the current basis."""
        basis = Basis(*args, parent=self, **kwargs)
        return basis

    def get_parents(self, include_root=True, include_self=False):
        """Get list of parent bases ordered from direct parent to root basis."""
        parents = [self] if include_self else []
        current = self
        while current.parent is not None:
            parents.append(current.parent)
            current = current.parent
        if not include_root:
            parents = parents[:-1]
        return parents

    def compatible(self, other: BasisType):
        return is_nobasis(other) or self.same_root(other)

    def find_common_parent(self, other):
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

    def _as_basis_matprod(self, other, simplify=False):
        """Return MatrixProduct required for as_basis method"""
        self.check_same_root(other)
        # Find first common ancestor and express coefficients in corresponding basis
        parent = self.find_common_parent(other.get_nondual())
        matprod = other.coeff_in_basis(parent).T + [parent.metric] + self.coeff_in_basis(parent)
        if simplify:
            matprod = matprod.simplify()
        return matprod

    def dual(self):
        return self._dual

    def get_nondual(self):
        return self

    @staticmethod
    def is_cobasis():
        return False

    @property
    def is_orthonormal(self):
        return isinstance(self.metric, IdentityMatrix)

    def is_derived_from(self, other, inclusive=False):
        """True if self is derived from other, else False"""
        if not self.same_root(other):
            return False
        for parent in self.get_parents(include_self=inclusive):
            if other == parent:
                return True
        return False

    def is_parent_of(self, other, inclusive=False):
        """True if self is parent of other, else False"""
        return other.is_derived_from(self, inclusive=inclusive)

    def get_orthonormal_error(self):
        ortherr = abs(self.metric-np.identity(self.size)).max()
        return ortherr

    def __neg__(self):
        return self.dual()


class Cobasis(BasisOrDualBasis):

    def __init__(self, basis, **kwargs):
        super().__init__(**kwargs)
        self._basis = basis

    @property
    def id(self):
        return -(self.dual().id)

    @property
    def size(self):
        return self.dual().size

    @property
    def variance(self) -> int:
        return -1

    def dual(self):
        return self._basis

    def get_nondual(self):
        return self.dual()

    @property
    def name(self):
        return 'Cobasis(%s)' % self.dual().name

    @staticmethod
    def is_dual():
        return True

    @property
    def root(self):
        return self.dual().root

    def coeff_in_basis(self, basis):
        matrices = self.get_nondual().coeff_in_basis(basis)
        # To raise right-hand index:
        matrices.append(self.metric)
        return matrices

    @property
    def metric(self):
        return self.dual().metric.inverse

    def _as_basis_matprod(self, other, simplify=False):
        """Append inverse metric (metric of dual space)"""
        matprod = self.get_nondual()._as_basis_matprod(other) + [self.metric]
        if simplify:
            matprod = matprod.simplify()
        return matprod

    def __pos__(self):
        return self.dual()

    @staticmethod
    def is_cobasis():
        return True


from .tensor import Tensor
