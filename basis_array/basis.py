import uuid
import numpy as np
from .util import *


def _get_overlap(coeff1, coeff2, metric=None):
    """Calculate the overlap overlap = (coeff1 | metric | coeff2)."""
    operands = [op for op in (coeff1.T, metric, coeff2) if not (isinstance(op, IdentityMatrix) or op is None)]
    if len(operands) == 0:
        assert coeff1.size == metric.size == coeff2.size
        return np.identity(coeff1.size)
    if len(operands) == 1:
        return operands[0]
    if len(operands) == 2:
        return np.dot(*operands)
    return np.linalg.multi_dot(operands)


class BasisBase:
    """Base class for Space and Basis class."""

    __next_id = 0

    def __init__(self, name=None, dual=False):
        #self.id = str(uuid.uuid4())
        self.id = BasisBase.__next_id
        BasisBase.__next_id += 1
        self.name = name
        self._dual = dual

    def make_basis(self, coeff=None, indices=None, **kwargs):
        """Make a new basis with coefficients or indices in reference to the current basis."""
        basis = Basis(self, coeff=coeff, indices=indices, **kwargs)
        return basis

    def __eq__(self, other):
        """Compare if to bases are the same based on their UUID."""
        if not isinstance(other, BasisBase):
            return False

        # Cannot compare bases in different spaces
        if self.root is not other.root:
            return False
            #raise ValueError("Cannot check identity of bases with different RootBasis.")
        return self.id == other.id

    def __repr__(self):
        name = (self.name or type(self).__name__)
        return '%s(size= %d)' % (name, self.size)
        #if self.name is None:
        #    return '%s(size= %d)' % (type(self).__name__, self.size)
        #return '%s(name= %s, size= %d)' % (self.name, type(self).__name__, self.size)

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
        return self.root == other.root

    def compatible(self, other):
        return other is nobasis or self.same_root(other)

    def check_same_root(self, other):
        if self.same_root(other):
            return
        raise BasisError("Bases %s and %s do not derive from the same root basis." % (self, other))

    def find_common_parent(self, other):
        """Find lowest common ancestor between two bases."""
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
        # Find lowest common ancestor and express coefficients in corresponding basis
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
        if not isinstance(other, BasisBase):
            return NotImplemented
        return other.as_basis(self)

    #def dual(self):
    #    if self.is_orthonormal:
    #        return self

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


class Space(BasisBase):
    """Space which has to be created before any other basis are defined.
    The space does not have any coefficients, only a size and, optionally, a metric."""

    def __init__(self, size, metric=None, name=None, dual=False):
        super().__init__(name=name, dual=dual)
        self.size = size
        if metric is None:
            metric = IdentityMatrix(size)
        self.metric = metric

    @property
    def parent(self):
        return None

    @property
    def root(self):
        return self

    def coeff_in_basis(self, basis):
        if basis is not self:
            raise ValueError
        return IdentityMatrix(self.size)


class Basis(BasisBase):
    """Basis class, requires a parent (another Basis or Space), coefficients or indices in terms of
    the parent basis."""

    #def __init__(self, parent, coeff=None, indices=None, orthonormal=None):
    def __init__(self, parent, rotation, orthonormal=None, name=None, dual=False):

        self.parent = parent
        super().__init__(name=name, dual=dual)

        if isinstance(rotation, int):
            if rotation >= self.parent.size:
                raise ValueError
            rotation = [rotation]
        # Convert to 2D-matrix (for the time being)
        if isinstance(rotation, (tuple, list, slice)) or (getattr(rotation, 'ndim', None) == 1):
            rotation = np.eye(self.parent.size)[:,rotation]
        elif isinstance(rotation, np.ndarray) and rotation.ndim == 2:
            pass
        else:
            raise ValueError("Invalid rotation: %r of type %r" % (rotation, type(rotation)))

        if rotation.shape[0] != self.parent.size:
            raise ValueError("Invalid size: %d (expected %d)" % (rotation.shape[0], self.parent.size))
        self.rotation = self.coeff = rotation

        # Calculate metric matrix for basis:
        self.metric = _get_overlap(rotation, rotation, metric=self.parent.metric)
        ortherr = self.get_orthonormal_error()
        if (orthonormal or ortherr < 1e-12):
            if ortherr > 1e-8:
                raise ValueError("Basis is not orthonormal:  error= %.3e" % ortherr)
            self.metric = IdentityMatrix(self.size)

    @property
    def size(self):
        """Number of basis functions."""
        return self.coeff.shape[1]

    @property
    def root(self):
        return self.parent.root

    def coeff_in_basis(self, basis):
        """Express coefficient matrix in the basis of another parent instead of the direct parent."""
        if basis.root != self.root:
            raise ValueError
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


from .array import Array
