import numpy as np
from functools import lru_cache
from btensor.util import *
from btensor.space import Space


def is_basis(obj, allow_int=True, allow_cobasis=True):
    allowed = (BasisClass,) if allow_cobasis else (Basis,)
    if allow_int:
        allowed += (int, np.integer)
    return isinstance(obj, allowed)


is_nobasis = is_int


def compatible_basis(b1, b2):
    if is_int(b1) and is_int(b2):
        if b1 == -1 or b2 == -1:
            return True
        return b1 == b2
    if is_int(b1):
        return b2.compatible(b1)
    return b1.compatible(b2)


class BasisClass:

    def __repr__(self):
        return '%s(id= %d, size= %d, name= %s)' % (type(self).__name__, self.id, self.size, self.name)

    def __eq__(self, other):
        """Compare if to bases are the same based on their ID."""
        if not isinstance(other, BasisClass):
            return False
        return hash(self) == hash(other)

    @property
    def id(self):
        raise NotImplementedError

    def __hash__(self):
        return self.id

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

    def __pos__(self):
        raise NotImplementedError

    def __neg__(self):
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
        if not isinstance(other, BasisClass):
            return NotImplemented
        return other.as_basis(self)

    def same_root(self, other):
        root1 = self.root or (+self)
        root2 = other.root or (+other)
        return root1 == root2

    def check_same_root(self, other):
        if not self.same_root(other):
            raise BasisError("Bases %s and %s do not derive from the same root basis." % (self, other))


class Basis(BasisClass):
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

    @property
    def id(self):
        return self._id

    @property
    def size(self):
        """Number of basis functions."""
        return self.coeff.shape[1]

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
        if not is_basis(basis, allow_int=False):
            raise ValueError

        if basis == self:
            return MatrixProduct([IdentityMatrix(self.size)])

        self.check_same_root(basis)
        parents = self.get_parents()
        nondual = +basis
        if nondual not in parents:
            raise ValueError("%s is not superbasis of %r" % (basis, self))
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

    def compatible(self, other):
        if is_int(other):
            return (other == self.size) or (other == -1)
        return self.same_root(other)

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
        parent = self.find_common_parent(+other)
        matprod = other.coeff_in_basis(parent).T + [parent.metric] + self.coeff_in_basis(parent)
        if simplify:
            matprod = matprod.simplify()
        return matprod

    def dual(self):
        return self._dual

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

    def __pos__(self):
        return self

    def __neg__(self):
        return self.dual()


class Cobasis(BasisClass):

    def __init__(self, basis, **kwargs):
        super().__init__(**kwargs)
        self._basis = basis

    @property
    def id(self):
        return -(self.dual().id)

    @property
    def size(self):
        return self.dual().size

    def dual(self):
        return self._basis

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
        matrices = (+self).coeff_in_basis(basis)
        # To raise right-hand index:
        matrices.append(self.metric)
        return matrices

    @property
    def metric(self):
        return self.dual().metric.inverse

    def _as_basis_matprod(self, other, simplify=False):
        """Append inverse metric (metric of dual space)"""
        matprod = (+self)._as_basis_matprod(other) + [self.metric]
        if simplify:
            matprod = matprod.simplify()
        return matprod

    def __pos__(self):
        return self.dual()

    def __neg__(self):
        return self


from .tensor import Tensor
