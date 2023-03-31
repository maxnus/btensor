"""Defines BasisArray class, which allows the attachment of bases to NumPy arrays.
The arrays can then be used in a custom einsum function, where the overlaps between
the different bases are automatically taken into account.

Usage:

>>> root = RootBasis(mol.nao, metric=mf.get_ovlp())
>>> mo = RootBasis.add_basis(mf.mo_coeff, name='mo')
>>> mo_occ = RootBasis.add_basis(mf.mo_coeff[:,occ], name='mo-occ')
>>> mo_vir = RootBasis.add_basis(mf.mo_coeff[:,vir], name='mo-vir')
>>> fov = BasisArray(fock[occ,vir], basis=(mo_occ, mo_vir))
>>> # View in different basis:
>>> print(fov.as_basis((mo, mo)))
>>> # Contract with other BasisArray:
>>> result = basis_einsum('ia,ja->ij', fov, t1)
>>> # The virtual dimension of `fov` and `t1` can be expressed in a different basis;
>>> # The corresponding overlap will be considered automatically.
>>> # `result` is another `BasisArray` instance.

"""

import numpy as np

from basis_array.util.util import nobasis
from .basis import RootBasis
from .basis import Basis
from .array import Array


A = Array


def B(rotation_or_size=None, parent=None, **kwargs):
    if parent is None:
        return RootBasis(rotation_or_size, **kwargs)
    return Basis(rotation_or_size, parent=parent, **kwargs)


from .numpy_functions import sum
from .numpy_functions import dot
from .numpy_functions import einsum
from .numpy_functions import linalg
