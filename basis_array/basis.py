import numpy as np
from basis_array.util import *
from basis_array.util.matrix import to_array
from basis_array.space import Space


class BasisClass:

    def __repr__(self):
        name = (self.name or type(self).__name__)
        return '%s(id= %d, size= %d)' % (name, self.id, self.size)

    def __eq__(self, other):
        """Compare if to bases are the same based on their ID."""
        if not isinstance(other, BasisClass):
            return False
        return self.id == other.id


class Basis(BasisClass):
    """Basis class.

    Parameters
    ----------
    a: Array, List, Int
    """

    __next_id = 1

    def __init__(self, a, parent=None, metric=None, orthonormal=None, name=None, dual=None, debug=False):
        self.parent = parent
        self._id = self._get_next_id()
        self.name = name
        self._dual = dual
        self.debug = debug or getattr(parent, 'debug', False)

        # Root basis:
        if isinstance(a, (int, np.integer)):
            a = IdentityMatrix(a)
        # Permutation + selection
        if isinstance(a, (tuple, list, slice)) or (getattr(a, 'ndim', None) == 1):
            #a = np.eye(self.parent.size)[:, a]
            a = PermutationMatrix(self.parent.size, a)
        elif isinstance(a, (np.ndarray, Matrix)) and a.ndim == 2:
            pass
        else:
            raise ValueError("Invalid rotation: %r of type %r" % (a, type(a)))

        if not self.is_root() and (a.shape[0] != self.parent.size):
            raise ValueError("Invalid size: %d (expected %d)" % (a.shape[0], self.parent.size))
        self._coeff = a

        # Calculate metric matrix for basis:
        if self.is_root():
            self.metric = metric if metric is not None else IdentityMatrix(self.size)
        else:
            if metric is not None:
                raise ValueError
            self.metric = chained_dot(a.T, self.parent.metric, a)
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
        """Express coefficient matrix in the basis of another parent instead of the direct parent."""
        self.check_same_root(basis)
        coeff = self.coeff
        if basis == self:
            return IdentityMatrix(self.size)
        for p in self.get_parents():
            if p == basis:
                break
            coeff = np.dot(p.coeff, coeff)
        else:
            raise RuntimeError
        return coeff

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
        parents = []
        current = self
        if include_self:
            parents.append(current)
        while current.parent is not None:
            parents.append(current.parent)
            current = current.parent
        if not include_root:
            parents = parents[:-1]
        return parents

    def same_root(self, other):
        root1 = self.root or self
        root2 = other.root or other
        return root1 == root2

    def compatible(self, other):
        return other is nobasis or self.same_root(other)

    def check_same_root(self, other):
        if not self.same_root(other):
            raise BasisError("Bases %s and %s do not derive from the same root basis." % (self, other))

    def find_common_parent(self, other):
        """Find first common ancestor between two bases."""
        self.check_same_root(other)
        parents1 = self.get_parents(include_self=True)[::-1]
        parents2 = other.get_parents(include_self=True)[::-1]
        assert (parents1[0] is parents2[0])
        for i, p in enumerate(parents1):
            if i >= len(parents2) or p != parents2[i]:
                break
            parent = p
        return parent

    def as_basis(self, other, metric=None):
        """Get overlap matrix as an Array with another basis."""
        self.check_same_root(other)
        # Find first common ancestor and express coefficients in corresponding basis
        parent = self.find_common_parent(other)

        #c_left = other.coeff_in_basis(parent)
        #c_right = self.coeff_in_basis(parent)
        #value = _get_overlap(c_left, c_right, metric=parent.metric)
        #if not other.is_orthonormal:
        #    m_left = other.metric
        #    value = np.dot(np.linalg.inv(m_left), value)

        matrices = [
            other.coeff_in_basis(parent).T,
            parent.metric,
            self.coeff_in_basis(parent)]
        if not other.is_orthonormal:
            matrices.insert(0, InverseMatrix(other.metric))
        value = chained_dot(*matrices)
        return Array(value, basis=(other, self), variance=(-1, 1))

    def __or__(self, other):
        """Allows writing overlap as `(basis1|basis2)`."""
        # other might still implement __ror__, so return NotImplemented instead of raising an exception
        if not isinstance(other, BasisClass):
            return NotImplemented
        return other.as_basis(self)

    def dual(self):
        if self.is_orthonormal:
            return self
        if self._dual is None:
            self._dual = DualBasis(self)
        return self._dual

    @property
    def is_orthonormal(self):
        return isinstance(self.metric, IdentityMatrix)

    def is_subbasis(self, other, inclusive=True):
        """True if other is subbasis of given basis, else False."""
        for parent in self.get_parents(include_self=inclusive):
            if other == parent:
                return True
        return False

    def is_superbasis(self, other, inclusive=True):
        """True if other is superbasis of given basis, else False."""
        return other.is_subbasis(self, inclusive=inclusive)

    def get_orthonormal_error(self):
        ortherr = abs(self.metric-np.identity(self.size)).max()
        return ortherr


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
        return 'dual(%s)' % self.dual().name


from .array import Array
