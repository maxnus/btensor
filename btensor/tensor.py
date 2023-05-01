import numbers
import string
import numpy as np
from btensor.util import *
from .basis import Basis, BasisClass, Cobasis
from .optemplate import OperatorTemplate
from . import numpy_functions


def value_if_scalar(array):
    if array.ndim > 0:
        return array
    return array._value


class Tensor(OperatorTemplate):
    """NumPy array with basis attached for each dimension."""

    def __init__(self, value, basis):
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
        self._value = np.asarray(value)
        self.basis = basis

    def __repr__(self):
        if np.all(self.variance_string == 1):
            return f'{type(self).__name__}(shape= {self.shape})'
        return f'{type(self).__name__}(shape= {self.shape}, variance= {self.variance})'

    def copy(self):
        return type(self)(self._value.copy(), basis=self.basis)

    # --- Basis

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
                raise ValueError("Basis instance or nobasis required.")
            if self.shape[i] != b.size:
                raise ValueError("Dimension %d with size %d incompatible with basis size %d" % (
                                 i+1, self.shape[i], b.size))
        if not hasattr(self, '_basis'):
            self._basis = value
        else:
            self.as_basis(value, inplace=True)
        #self.project_onto(value, inplace=True)

    def replace_basis(self, basis, inplace=False):
        """Replace basis with new basis."""
        if basis is nobasis or isinstance(basis, BasisClass):
            basis = (basis,)
        new_basis = list(self.basis)
        for i, (b0, b1) in enumerate(zip(self.basis, basis)):
            if b1 is None:
                continue
            if (b1 is not nobasis) and (b1.size != self.shape[i]):
                raise ValueError("Dimension %d with size %d incompatible with basis size %d" % (
                                 i+1, self.shape[i], b1.size))
            new_basis[i] = b1
        assert len(new_basis) == len(self.basis)
        tensor = self if inplace else self.copy()
        tensor._basis = tuple(new_basis)
        return tensor

    # --- Variance

    @property
    def variance(self):
        return tuple([1 if isinstance(b, Cobasis) else -1 for b in self.basis])

    def as_variance(self, variance):
        if np.ndim(variance) == 0:
            variance = self.ndim * (variance,)
        if len(variance) != self.ndim:
            raise ValueError("%d-dimensional Array requires %d variance elements (%d given)" % (
                             self.ndim, self.ndim, len(variance)))
        if not np.isin(variance, (-1, 1)):
            raise ValueError("Variance can only contain values -1 and 1")
        new_basis = []
        for i, (b, v0, v1) in enumerate(zip(self.basis, self.variance, variance)):
            if v0 != v1:
                b = b.dual()
            new_basis.append(b)
        return self.as_basis(basis=new_basis)

    @property
    def covariant_axes(self):
        return tuple(np.asarray(self.variance) == 1)

    @property
    def contravariant_axes(self):
        return tuple(np.asarray(self.variance) == -1)

    @property
    def variance_string(self):
        """String representation of variance tuple."""
        symbols = {1: '+', -1: '-', 0: '*'}
        return ''.join(symbols[x] for x in self.variance)

    # --- NumPy compatibility

    def to_array(self, basis=None):
        """Convert to NumPy ndarray"""
        tensor = self.as_basis(basis=basis) if basis is not None else self
        return tensor._value.copy()

    def __getattr__(self, name):
        """Inherit from NumPy"""
        if name in ['dtype', 'ndim', 'shape', '__array_interface__']:
            return getattr(self._value, name)
        raise AttributeError("%r object has no attribute '%s'" % (self.__class__.__name__, name))

    def __getitem__(self, key):
        """Construct and return sub-Array."""
        if isinstance(key, int):
            return type(self)(self._value[key], basis=self.basis[1:])
        if isinstance(key, (list, np.ndarray)):
            value = self._value[key]
            basis = (self.basis[0].make_basis(key),) + self.basis[1:]
            return type(self)(value, basis=basis)
        if key is Ellipsis:
            return self
        if isinstance(key, slice) or key is np.newaxis:
            key = (key,)
        if isinstance(key, tuple):
            value = self._value[key]
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
                    basis[idx] = Basis(argument=ki, parent=basis[idx])
                elif ki is np.newaxis:
                    pass
                else:
                    raise ValueError("key %r of type %r" % (ki, type(ki)))
            basis = tuple(basis)
            return type(self)(value, basis=basis)
        raise NotImplementedError("Key= %r of type %r" % (key, type(key)))

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value._value
        self._value[key] = value
        # Not required, since np.newaxis has no effect in assignment?
        #if not isinstance(key, tuple) or np.newaxis not in key:
        #    return
        #basis_old = list(self.basis)
        #basis_new = tuple(nobasis if elem is np.newaxis else basis_old.pop(0) for elem in key)
        #self.basis = basis_new

    def transpose(self, axes=None):
        value = self._value.transpose(axes)
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

    @staticmethod
    def _get_basis_transform(basis1, basis2):
        # Avoid evaluating the overlap, if not necessary (e.g. for a permutation matrix)
        # transform = (basis1 | basis2.dual()).value
        transform = basis2.dual()._as_basis_matprod(basis1, simplify=True)
        return transform

    def proj(self, basis, inplace=False):
        """Transform to different set of basis.

        None can be used to indicate no transformation.

        Note that this can reduce the rank of the array, for example when trying to express
        a purely occupied quantitiy in a purely virtual basis.
        """
        basis = self._broadcast_basis(basis)
        assert len(basis) == len(self.basis)
        for bas in basis:
            if not (isinstance(bas, BasisClass) or bas in (nobasis, None)):
                raise ValueError

        subscripts = string.ascii_lowercase[:self.ndim]
        operands = [self._value]
        result = list(subscripts)
        basis_out = list(self.basis)
        for i, bas in enumerate(basis):
            if bas is None or (bas == self.basis[i]):
                #if bas is None:
                #    basis_out[i] = self.basis[i]
                continue
            # Remove basis:
            if bas is nobasis:
                basis_out[i] = nobasis
                continue

            basis_out[i] = bas
            # Add basis (buggy):
            if self.basis[i] is nobasis:
                continue

            # Avoid evaluating the overlap, if not necessary (e.g. for a permutation matrix)
            #ovlp = (self.basis[i] | bas.dual()).value
            #ovlp = bas.dual()._as_basis_matprod(self.basis[i], simplify=True)
            ovlp = self._get_basis_transform(self.basis[i], bas)
            if len(ovlp) == 1 and isinstance(ovlp[0], IdentityMatrix):
                raise NotImplementedError
            elif len(ovlp) == 1 and isinstance(ovlp[0], PermutationMatrix):
                perm = ovlp[0]
                indices = perm.indices
                if isinstance(perm, RowPermutationMatrix):
                    operands[0] = util.expand_axis(operands[0], perm.shape[1], indices=indices, axis=i)
                if isinstance(perm, ColumnPermutationMatrix):
                    operands[0] = np.take(operands[0], indices, axis=i)
                continue
            else:
                ovlp = ovlp.evaluate()
            operands.append(ovlp)
            sub_new = subscripts[i].upper()
            subscripts += (',%s%s' % (subscripts[i], sub_new))
            result[i] = sub_new
        basis_out = tuple(basis_out)
        subscripts += '->%s' % (''.join(result))
        value = np.einsum(subscripts, *operands, optimize=True)
        if inplace:
            self._value = value
            self._basis = basis_out
            return self
        return type(self)(value, basis=basis_out)

    def as_basis(self, basis, inplace=False):
        if isinstance(basis, BasisClass):
            basis = (basis,)
        for b0, b1 in zip(self.basis, basis):
            if b1 is None:
                continue
            b0 = +b0
            b1 = +b1
            if not (b1.space >= b0.space):
                raise BasisError(f"{b1} does not span {b0}")
        return self.proj(basis, inplace=inplace)

    def as_basis_at(self, index, basis, **kwargs):
        if index < 0:
            index += self.ndim
        basis_new = self.basis[:index] + (basis,) + self.basis[index+1:]
        return self.as_basis(basis_new, **kwargs)

    def __or__(self, basis):
        """To allow basis transformation as (array | basis)"""
        # Left-pad:
        basis = self._broadcast_basis(basis, pad='left')
        return self.as_basis(basis)

    def __ror__(self, basis):
        """To allow basis transformation as (basis | array)"""
        return self.as_basis(basis)

    def _broadcast_basis(self, basis, pad='right'):
        """Broadcast basis to same length as self.basis."""
        if isinstance(basis, BasisClass):
            basis = (basis,)
        npad = len(self.basis) - len(basis)
        if npad == 0:
            return tuple(basis)
        if npad < 0:
            raise ValueError
        if pad == 'right':
            return tuple(basis) + npad*(None,)
        if pad == 'left':
            return npad*(None,) + tuple(basis)
        raise ValueError

    # Arithmetic

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
            if b1.is_cobasis() ^ b2.is_cobasis():
                raise ValueError()
            if b1 is nobasis and b2 is nobasis:
                basis.append(nobasis)
            elif b1 is nobasis:
                basis.append(b2)
            elif b2 is nobasis:
                basis.append(b1)
            else:
                basis.append(b1.find_common_parent(b2))
        return tuple(basis)

    def _operator(self, operator, *other, swap=False):
        # Unary operator
        if len(other) == 0:
            return type(self)(operator(self._value), basis=self.basis)
        # Ternary+ operator
        if len(other) > 1:
            raise NotImplementedError
        # Binary operator
        other = other[0]

        basis = self.basis
        v1 = self._value
        if isinstance(other, numbers.Number):
            v2 = other
        elif isinstance(other, Tensor):
            if self.basis == other.basis:
                v2 = other._value
            elif self.is_compatible(other):
                basis = self.common_basis(other)
                v1 = self.as_basis(basis)._value
                v2 = other.as_basis(basis)._value
            else:
                raise ValueError
        else:
            return NotImplemented
        if swap:
            v1, v2 = v2, v1
        return type(self)(operator(v1, v2), basis=basis)


class Cotensor(Tensor):

    @property
    def variance(self):
        return tuple([-1 if isinstance(b, Cobasis) else 1 for b in self.basis])

    @staticmethod
    def _get_basis_transform(basis1, basis2):
        transform = basis2._as_basis_matprod(basis1.dual(), simplify=True)
        return transform