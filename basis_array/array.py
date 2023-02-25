import string
import numpy as np
from .util import *
from .basis import BasisBase


class Array:
    """NumPy array with bases attached for each dimension."""

    def __init__(self, value, basis, contravariant=False):
        if len(basis) != np.ndim(value):
            raise ValueError("Array with shape %r requires %d bases, %d given" % (
                value.shape, np.ndim(value), len(basis)))
        for i, b in enumerate(basis):
            if value.shape[i] != b.size:
                raise ValueError("Dimension %d with size %d incompatible with basis size %d" % (
                    i+1, value.shape[i], b.size))
        self.value = value
        self.basis = basis
        if np.ndim(contravariant) == 0:
            contravariant = self.ndim * [contravariant]
        self.contravariant = contravariant

    def __repr__(self):
        return '%s(shape= %r)' % (self.__class__.__name__, self.shape)

    def __getattr__(self, name):
        if name in ['dtype', 'ndim', 'shape', '__array_interface__']:
            return getattr(self.value, name)
        raise AttributeError("%r object has no attribute '%s'" % (self.__class__.__name__, name))

    def __getitem__(self, key):
        """Allow direct access of array data as array[key]."""
        return self.value[key]

    def as_basis(self, basis):
        """Transform to different set of bases.

        None can be used to indicate no transformation.

        Note that this can reduce the rank of the array, for example when trying to express
        a purely occupied quantitiy in a purely virtual basis.
        """
        if len(basis) != len(self.basis):
            raise ValueError
        for bas in basis:
            if not isinstance(bas, (BasisBase, type(None))):
                raise ValueError

        subscripts = string.ascii_lowercase[:self.ndim]
        operands = [self.value]
        result = ''
        basis_out = list(basis)
        for i, bas in enumerate(basis):
            if bas is None or (bas == self.basis[i]):
                result += subscripts[i]
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
        subscripts += '->%s' % (''.join(result))
        value = np.einsum(subscripts, *operands, optimize=True)
        return Array(value, tuple(basis_out), contravariant=self.contravariant)

    def __rshift__(self, basis):
        """To allow basis transformation as array >> basis"""
        return self.__or__(basis)

    def __or__(self, basis):
        """To allow basis transformation as (array | basis)"""
        if isinstance(basis, BasisBase):
            basis = (basis,)
        if isinstance(basis, tuple):
            basis = self.basis[:-len(basis)] + basis
        return self.as_basis(basis)

    def __rlshift__(self, basis):
        """To allow basis transformation as basis << array"""
        return self.__ror__(basis)

    def __ror__(self, basis):
        """To allow basis transformation as (basis | array)"""
        if isinstance(basis, BasisBase):
            basis = (basis,)
        if isinstance(basis, tuple):
            basis = basis + self.basis[len(basis):]
        return self.as_basis(basis)
