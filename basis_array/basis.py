import numpy as np
from basis_array.util import *


class BasisClass:

    def __repr__(self):
        name = (self.name or type(self).__name__)
        return '%s(id= %d, size= %d)' % (name, self.id, self.size)

    def __eq__(self, other):
        """Compare if to bases are the same based on their ID."""
        if not isinstance(other, BasisClass):
            return False
        return self.id == other.id

    def dual(self):
        raise NotImplementedError

    def __invert__(self):
        return self.dual()

    def get_nondual(self):
        raise NotImplementedError

    def as_basis(self, other):
        raise NotImplementedError

    def __or__(self, other):
        """Allows writing overlap as `(basis1 | basis2)`."""
        # other might still implement __ror__, so return NotImplemented instead of raising an exception
        if not isinstance(other, BasisClass):
            return NotImplemented
        return other.as_basis(self)

    def same_root(self, other):
        root1 = self.root or self.get_nondual()
        root2 = other.root or other.get_nondual()
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

    def __init__(self, argument, parent=None, metric=None, orthonormal=None, name=None, dual=None, debug=False):
        self.parent = parent
        self._id = self._get_next_id()
        self.name = name
        self._dual = dual
        self.debug = debug or getattr(parent, 'debug', False)

        # Root basis:
        if isinstance(argument, (int, np.integer)):
            argument = IdentityMatrix(argument)
        # Permutation + selection
        if isinstance(argument, (tuple, list, slice)) or (getattr(argument, 'ndim', None) == 1):
            argument = ColumnPermutationMatrix(self.parent.size, argument)
        elif isinstance(argument, (np.ndarray, Matrix)) and argument.ndim == 2:
            pass
        else:
            raise ValueError("Invalid rotation: %r of type %r" % (argument, type(argument)))

        if not self.is_root() and (argument.shape[0] != self.parent.size):
            raise ValueError("Invalid size: %d (expected %d)" % (argument.shape[0], self.parent.size))
        self._coeff = argument

        # Calculate metric matrix for basis:
        if self.is_root():
            self.metric = metric if metric is not None else IdentityMatrix(self.size)
        else:
            if metric is not None:
                raise ValueError
            self.metric = MatrixProduct((argument.T, self.parent.metric, argument)).evaluate()
            if self.debug or __debug__:
                cond = np.linalg.cond(self.metric)
                if cond > 1e14:
                    raise RuntimeError("Large condition number of metric matrix: %e" % cond)
            ortherr = self.get_orthonormal_error()
            if (orthonormal or ortherr < 1e-12):
                if ortherr > 1e-8:
                    raise ValueError("Basis is not orthonormal:  error= %.3e" % ortherr)
                self.metric = IdentityMatrix(self.size)

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
    def root(self):
        if self.parent is None:
            return None
        if self.parent.root is None:
            return self.parent
        return self.parent.root

    def is_root(self):
        return self.parent is None

    def coeff_in_basis(self, basis):
        """Express coeffients in different (parent) basis (rather than the direct parent).

        Was BUGGY before, now fixed?"""
        if basis == self:
            return [IdentityMatrix(self.size)]

        self.check_same_root(basis)
        parents = self.get_parents()
        nondual = basis.get_nondual()
        if nondual not in parents:
            raise ValueError("%s is not superbasis of %r" % (basis, self))
        matrices = []
        for p in parents:
            if p == nondual:
                break
            matrices.append(p.coeff)

        if basis.is_dual():
            raise NotImplementedError
        #    print(matrices[-1].shape)
        #    print(basis.metric.shape)
            #matrices.append(basis.metric)

        matrices = matrices[::-1]
        matrices.append(self.coeff)

        #if basis.is_dual():
        #    print(matrices[-1].shape)
        #    print(basis.metric.shape)
        #    matrices.append(basis.metric)
        #matrices = [to_array(x) for x in matrices]

        return matrices

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
        return other is nobasis or self.same_root(other)

    def find_common_parent(self, other):
        """Find first common ancestor between two bases."""
        a = self
        b = other
        #a = self.get_nondual()
        #b = other.get_nondual()
        a.check_same_root(b)
        parents1 = a.get_parents(include_self=True)[::-1]
        parents2 = b.get_parents(include_self=True)[::-1]
        assert (parents1[0] is parents2[0])
        for i, p in enumerate(parents1):
            if i >= len(parents2) or p != parents2[i]:
                break
            parent = p
        return parent

    def as_basis(self, other):
        """Get overlap matrix as an Array with another basis."""
        self.check_same_root(other)
        # Find first common ancestor and express coefficients in corresponding basis
        parent = self.find_common_parent(other.get_nondual())

        matrices = [x.T for x in other.coeff_in_basis(parent)]
        matrices.append(parent.metric)
        matrices.extend(self.coeff_in_basis(parent))
        # This is now done in the DualBasis
        #if other.is_dual():
        #    matrices.insert(0, other.metric)
        value = MatrixProduct(matrices).evaluate()
        return Array(value, basis=(other, self), variance=(-1, 1))

    def dual(self):
        if self.is_orthonormal:
            return self
        if self._dual is None:
            self._dual = DualBasis(self)
        return self._dual

    @staticmethod
    def is_dual():
        return False

    @property
    def is_orthonormal(self):
        return isinstance(self.metric, IdentityMatrix)

    #def is_subbasis(self, other, inclusive=True):
    #    """True if other is subbasis of given basis, else False."""
    #    for parent in self.get_parents(include_self=inclusive):
    #        if other == parent:
    #            return True
    #    return False

    #def is_superbasis(self, other, inclusive=True):
    #    """True if other is superbasis of given basis, else False."""
    #    return other.is_subbasis(self, inclusive=inclusive)

    def get_orthonormal_error(self):
        ortherr = abs(self.metric-np.identity(self.size)).max()
        return ortherr

    def get_nondual(self):
        return self


class DualBasis(BasisClass):

    def __init__(self, basis):
        self._basis = basis

    @property
    def id(self):
        return -self.dual().id

    @property
    def size(self):
        return self.dual().size

    def dual(self):
        return self._basis

    @property
    def name(self):
        return 'Dual(%s)' % self.dual().name

    @staticmethod
    def is_dual():
        return True

    def get_nondual(self):
        return self.dual()

    @property
    def root(self):
        return self.dual().root

    #def get_parents(self, include_root=True, include_self=False):
    #    return self.dual().get_parents(include_root=include_root, include_self=include_self)

    def coeff_in_basis(self, basis):
        matrices = self.get_nondual().coeff_in_basis(basis)
        matrices.append(self.metric)
        return matrices

    @property
    def metric(self):
        return InverseMatrix(self.dual().metric)

    def as_basis(self, other):
        c = self.dual().as_basis(other)
        value = MatrixProduct((c.value, self.metric)).evaluate()
        return Array(value, basis=(other, self))

from .array import Array
