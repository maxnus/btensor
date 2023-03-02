import operator
import string
import itertools
import copy
import numpy as np
from .util import *
from .basis import Basis, BasisBase, NoBasis
from .optemplate import OperatorTemplate
from . import numpy_functions


def value_if_scalar(array):
    if array.ndim > 0:
        return array
    return array.value


class Array(OperatorTemplate):
    """NumPy array with bases attached for each dimension."""

    def __init__(self, value, basis, contravariant=False):
        if basis is NoBasis or isinstance(basis, BasisBase):
            basis = (basis,)
        if len(basis) != np.ndim(value):
            raise ValueError("Array with shape %r requires %d bases, %d given" % (
                value.shape, np.ndim(value), len(basis)))
        for i, b in enumerate(basis):
            if b is NoBasis:
                continue
            if not isinstance(b, BasisBase):
                raise ValueError("Basis instance or None required")
            if value.shape[i] != b.size:
                raise ValueError("Dimension %d with size %d incompatible with basis size %d" % (
                    i+1, value.shape[i], b.size))
        self.value = value
        self._basis = basis
        if np.ndim(contravariant) == 0:
            contravariant = self.ndim * [contravariant]
        self.contravariant = contravariant

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, value):
        self.as_basis(value, inplace=True)

    def copy(self):
        return type(self)(self.value.copy(), basis=self.basis)

    def __repr__(self):
        return '%s(shape= %r)' % (self.__class__.__name__, self.shape)

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
        if isinstance(key, slice) or key is None:
            key = (key,)
        if isinstance(key, tuple):
            value = self.value[key]
            if value.ndim == 0:
                return value

            # Add NoBasis for each newaxis (None) key
            newaxis_indices = [i for (i, k) in enumerate(key) if (k is np.newaxis)]
            basis = list(self.basis)
            for i in newaxis_indices:
                basis.insert(i, NoBasis)

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
                    basis[idx] = Basis(basis[idx], rotation=ki)
                elif ki is None:
                    pass
                else:
                    raise ValueError("key %r of type %r" % (ki, type(ki)))
            basis = tuple(basis)
            return type(self)(value, basis=basis)
        raise NotImplementedError("Key= %r of type %r" % (key, type(key)))

    def transpose(self, axes=None):
        values = self.value.transpose(axes)
        if axes is None:
            basis = self.basis[::-1]
        else:
            basis = tuple(self.basis[ax] for ax in axes)
        return type(self)(values, basis=basis)

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

    def as_basis(self, basis, inplace=False):
        """Transform to different set of basis.

        None can be used to indicate no transformation.

        Note that this can reduce the rank of the array, for example when trying to express
        a purely occupied quantitiy in a purely virtual basis.
        """
        if len(basis) != len(self.basis):
            raise ValueError
        for bas in basis:
            if not (isinstance(bas, BasisBase) or bas is NoBasis):
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
            # If self.basis[i] is covariant and bas is contravariant (or vice versa), the order
            # of bases in the overlap matters:
            if not self.contravariant[i]:
                ovlp = (self.basis[i] | bas).value
            else:
                ovlp = (bas | self.basis[i]).value.T
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
        return type(self)(value, basis=basis, contravariant=self.contravariant)

    def as_basis_at(self, index, basis, **kwargs):
        if index < 0:
            index += self.ndim
        basis_new = self.basis[:index] + (basis,) + self.basis[index+1:]
        return self.as_basis(basis_new, **kwargs)


    #def __rshift__(self, basis):
    #    """To allow basis transformation as array >> basis"""
    #    return self.__or__(basis)

    #def __rlshift__(self, basis):
    #    """To allow basis transformation as basis << array"""
    #    return self.__ror__(basis)

    def __or__(self, basis):
        """To allow basis transformation as (array | basis)"""
        if isinstance(basis, BasisBase):
            basis = (basis,)
        if isinstance(basis, tuple):
            basis = self.basis[:-len(basis)] + basis
        return self.as_basis(basis)

    def __ror__(self, basis):
        """To allow basis transformation as (basis | array)"""
        if isinstance(basis, BasisBase):
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
            if (b1 is None or b2 is None) and (self.shape[i] == other.shape[i]):
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
            if b1 is NoBasis and b2 is NoBasis:
                basis.append(NoBasis)
            elif b1 is NoBasis:
                basis.append(b2)
            elif b2 is NoBasis:
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
