import string
import numpy as np
from basis_array.util import *
from .basis import Basis, BasisClass
from .optemplate import OperatorTemplate
from . import numpy_functions


def value_if_scalar(array):
    if array.ndim > 0:
        return array
    return array.value


class Array(OperatorTemplate):
    """NumPy array with basis attached for each dimension."""

    def __init__(self, value, basis, variance=1):
        #if basis is nobasis or isinstance(basis, BasisBase):
        #    basis = (basis,)
        #if len(basis) != np.ndim(value):
        #    raise ValueError("Array with shape %r requires %d bases, %d given" % (
        #        value.shape, np.ndim(value), len(basis)))
        #for i, b in enumerate(basis):
        #    if b is nobasis:
        #        continue
        #    if not isinstance(b, BasisBase):
        #        raise ValueError("Basis instance or nobasis required")
        #    if value.shape[i] != b.size:
        #        raise ValueError("Dimension %d with size %d incompatible with basis size %d" % (
        #            i+1, value.shape[i], b.size))
        self.value = value
        self.basis = basis
        self.replace_variance(variance)

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, value):
        if value is nobasis or isinstance(value, BasisClass):
            value = (value,)
        if len(value) != self.ndim:
            raise ValueError("%d-dimensional Array requires %d basis elements (%d given)" % (
                             self.ndim, self.ndim, len(value)))
        for i, b in enumerate(value):
            if b is nobasis:
                continue
            if not isinstance(b, BasisClass):
                raise ValueError("Basis instance or nobasis required")
            if self.shape[i] != b.size:
                raise ValueError("Dimension %d with size %d incompatible with basis size %d" % (
                                 i+1, self.shape[i], b.size))
        if not hasattr(self, '_basis'):
            self._basis = value
        else:
            self.as_basis(value, inplace=True)
        #self.project_onto(value, inplace=True)

    def replace_basis(self, basis):
        """Replace basis with new basis."""
        if basis is nobasis or isinstance(basis, BasisClass):
            value = (basis,)
        new_basis = list(self.basis)
        for i, (b0, b1) in enumerate(zip(self.basis, basis)):
            if b1 is None:
                continue
            if (b1 is not nobasis) and (b1.size != self.shape[i]):
                raise ValueError("Dimension %d with size %d incompatible with basis size %d" % (
                                 i+1, self.shape[i], b1.size))
            new_basis[i] = b1
        assert len(new_basis) == len(self.basis)
        self._basis = tuple(new_basis)

    @property
    def variance(self):
        raise NotImplementedError
        return self._variance

    #@variance.setter
    #def variance(self, value):
    #    if np.ndim(value) == 0:
    #        value = self.ndim * (value,)
    #    if len(value) != self.ndim:
    #        raise ValueError("%d-dimensional Array requires %d variance elements (%d given)" % (
    #                         self.ndim, self.ndim, len(value)))
    #    self._variance = value

    def _check_variance(self, variance):
        if np.ndim(variance) == 0:
            variance = self.ndim * (variance,)
        if len(variance) != self.ndim:
            raise ValueError("%d-dimensional Array requires %d variance elements (%d given)" % (
                             self.ndim, self.ndim, len(variance)))
        return variance

    def replace_variance(self, variance):
        self._variance = self._check_variance(variance)

    #def as_variance(self, variance):
    #    variance = self._check_variance(variance)
    #    for i, (v0, v1) in enumerate(zip(self.variance, variance)):
    #        if v0 == v1:
    #            continue
    #    return type(self)(value, basis=basis, variance=variance)

    @property
    def covariant_axes(self):
        return tuple(np.asarray(self.variance) == 1)

    @property
    def contravariant_axes(self):
        return tuple(np.asarray(self.variance) == -1)

    def copy(self):
        return type(self)(self.value.copy(), basis=self.basis, variance=self.variance)

    @property
    def variance_string(self):
        """String representation of variance tuple."""
        symbols = {1: '+', -1: '-', 0: '*'}
        return ''.join(symbols[x] for x in self.variance)

    def __repr__(self):
        return '%s(shape= %r, variance= %r)' % (self.__class__.__name__, self.shape, self.variance_string)

    # --- NumPy compatibility

    def __getattr__(self, name):
        """Inherit from NumPy"""
        if name in ['dtype', 'ndim', 'shape', '__array_interface__']:
            return getattr(self.value, name)
        raise AttributeError("%r object has no attribute '%s'" % (self.__class__.__name__, name))

    def __getitem__(self, key):
        """Construct and return sub-Array."""
        if isinstance(key, int):
            return type(self)(self.value[key], basis=self.basis[1:])
        if key is Ellipsis:
            return self
        if isinstance(key, slice) or key is np.newaxis:
            key = (key,)
        if isinstance(key, tuple):
            value = self.value[key]
            if value.ndim == 0:
                return value

            # Add nobasis for each newaxis (None) key
            newaxis_indices = [i for (i, k) in enumerate(key) if (k is np.newaxis)]
            basis = list(self.basis)
            for i in newaxis_indices:
                basis.insert(i, nobasis)

            # Replace Ellipsis with multiple slice(None)
            if Ellipsis in key:
                idx = key.index(Ellipsis)
                ellipsis_size = len(basis) - len(key) + 1
                key = key[:idx] + ellipsis_size*(slice(None),) + key[idx+1:]

            for i, ki in enumerate(reversed(key), start=1):
                idx = len(key) - i
                if isinstance(ki, (int, np.integer)):
                    del basis[idx]
                elif isinstance(ki, slice):
                    basis[idx] = Basis(a=ki, parent=basis[idx])
                elif ki is np.newaxis:
                    pass
                else:
                    raise ValueError("key %r of type %r" % (ki, type(ki)))
            basis = tuple(basis)
            return type(self)(value, basis=basis)
        raise NotImplementedError("Key= %r of type %r" % (key, type(key)))

    def transpose(self, axes=None):
        value = self.value.transpose(axes)
        if axes is None:
            basis = self.basis[::-1]
        else:
            basis = tuple(self.basis[ax] for ax in axes)
        return type(self)(value, basis=basis)

    @property
    def T(self):
        return self.transpose()

    def sum(self, axis=None):
        return numpy_functions.sum(self, axis=axis)

    def trace(self, axis1=0, axis2=1):
        return numpy_functions.trace(self, axis1=axis1, axis2=axis2)

    def dot(self, b):
        return numpy_functions.dot(self, b)

    # ---

    def index_non_subbasis(self, basis, inclusive=True):
        """Index of first element of basis which is not a subbasis of the corresponding array basis element."""
        for i, (b0, b1) in enumerate(zip(self.basis, basis)):
            if not b1.is_subbasis(b0, inclusive=inclusive):
                return i
        return -1

    def index_non_superbasis(self, basis, inclusive=True):
        """Index of first element of basis which is not a superbasis of the corresponding array basis element."""
        for i, (b0, b1) in enumerate(zip(self.basis, basis)):
            if not b1.is_superbasis(b0, inclusive=inclusive):
                return i
        return -1

    def is_subbasis(self, basis, inclusive=True):
        return (self.index_non_subbasis(basis, inclusive=inclusive) == -1)

    def is_superbasis(self, basis, inclusive=True):
        return (self.index_non_superbasis(basis, inclusive=inclusive) == -1)

    def has_subbasis(self, other, inclusive=True):
        return self.is_subbasis(other.basis, inclusive=inclusive)

    def has_superbasis(self, other, inclusive=True):
        return self.is_superbasis(other.basis, inclusive=inclusive)

    def as_basis(self, basis, inplace=False):
        #i = self.index_non_superbasis(basis)
        #if i != -1:
        #    raise BasisError("%s is not superbasis of %s" % (basis[i], self.basis[i]))
        return self.project_onto(basis, inplace=inplace)

    def project_onto(self, basis, inplace=False):
        """Transform to different set of basis.

        None can be used to indicate no transformation.

        Note that this can reduce the rank of the array, for example when trying to express
        a purely occupied quantitiy in a purely virtual basis.
        """
        if len(basis) != len(self.basis):
            raise ValueError
        for bas in basis:
            if not (isinstance(bas, BasisClass) or bas is nobasis):
                raise ValueError

        subscripts = string.ascii_lowercase[:self.ndim]
        operands = [self.value]
        result = ''
        basis_out = list(basis)
        for i, bas in enumerate(basis):
            if bas is None or (bas == self.basis[i]):
                result += subscripts[i]
                #if bas is None:
                if bas is None:
                    basis_out[i] = self.basis[i]
                continue
            # Remove basis:
            if bas is nobasis:
                result += subscripts[i]
                basis_out[i] = nobasis
                continue
            # Add basis:
            # Buggy
            if self.basis[i] is nobasis:
                result += subscripts[i]
                basis_out[i] = bas
                continue

            # If self.basis[i] is covariant and bas is contravariant (or vice versa), the order
            # of bases in the overlap matters:
            #if not self.contravariant_axes[i]:
            #    ovlp = (self.basis[i] | bas).value
            #else:
            #    ovlp = (bas | self.basis[i]).value.T
            ovlp = (~self.basis[i] | bas).value
            operands.append(ovlp)
            sub_new = subscripts[i].upper()
            subscripts += (',%s%s' % (subscripts[i], sub_new))
            result += sub_new
        basis_out = tuple(basis_out)
        subscripts += '->%s' % (''.join(result))
        value = np.einsum(subscripts, *operands, optimize=True)
        if inplace:
            self.value = value
            self._basis = basis_out
            return self
        return type(self)(value, basis=basis)#, variance=self.variance)

    def as_basis_at(self, index, basis, **kwargs):
        if index < 0:
            index += self.ndim
        basis_new = self.basis[:index] + (basis,) + self.basis[index+1:]
        return self.as_basis(basis_new, **kwargs)

    def __or__(self, basis):
        """To allow basis transformation as (array | basis)"""
        if isinstance(basis, BasisClass):
            basis = (basis,)
        if isinstance(basis, tuple):
            basis = self.basis[:-len(basis)] + basis
        return self.as_basis(basis)

    def __ror__(self, basis):
        """To allow basis transformation as (basis | array)"""
        if isinstance(basis, BasisClass):
            basis = (basis,)
        if isinstance(basis, tuple):
            basis = basis + self.basis[len(basis):]
        return self.as_basis(basis)

    # Arithmetric

    def is_compatible(self, other):
        return all(self.compatible_axes(other))

    def compatible_axes(self, other):
        axes = []
        for i, (b1, b2) in enumerate(zip(self.basis, other.basis)):
            if (b1 is nobasis or b2 is nobasis) and (self.shape[i] == other.shape[i]):
                axes.append(True)
            elif b1.compatible(b2):
                axes.append(True)
            else:
                axes.append(False)
        if self.ndim > other.ndim:
            axes += (self.ndim-other.ndim)*[False]
        return axes

    def common_basis(self, other):
        basis = []
        if not self.is_compatible(other):
            raise ValueError
        for b1, b2 in zip(self.basis, other.basis):
            if b1 is nobasis and b2 is nobasis:
                basis.append(nobasis)
            elif b1 is nobasis:
                basis.append(b2)
            elif b2 is nobasis:
                basis.append(b1)
            else:
                basis.append(b1.find_common_parent(b2))
        return tuple(basis)

    def _operator(self, operator, *other):
        # Unary operator
        if len(other) == 0:
            return type(self)(operator(self.value), basis=self.basis)
        # Ternary+ operator
        if len(other) > 1:
            raise NotImplementedError
        # Binary operator
        other = other[0]
        if (self.basis == other.basis):
            basis = self.basis
            v1 = self.value
            v2 = other.value
        elif self.is_compatible(other):
            basis = self.common_basis(other)
            v1 = self.as_basis(basis).value
            v2 = other.as_basis(basis).value
        else:
            raise ValueError
        return type(self)(operator(v1, v2), basis=basis)
