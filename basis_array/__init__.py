"""Defines BasisArray class, which allows the attachment of bases to NumPy arrays.
The arrays can then be used in a custom einsum function, where the overlaps between
the different bases are automatically taken into account.

Usage:

>>> root = Space(mol.nao, metric=mf.get_ovlp())
>>> mo = Space.add_basis(mf.mo_coeff, name='mo')
>>> mo_occ = Space.add_basis(mf.mo_coeff[:,occ], name='mo-occ')
>>> mo_vir = Space.add_basis(mf.mo_coeff[:,vir], name='mo-vir')
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

from .util import nobasis
from .basis import Basis, RootBasis
from .array import Array

A = Array

def B(parent_or_size, rotation=None, **kwargs):
    if rotation is None:
        if not isinstance(parent_or_size, (int, np.integer)):
            raise ValueError
        return RootBasis(parent_or_size, **kwargs)
    return Basis(parent_or_size, rotation=rotation, **kwargs)

from .numpy_functions import sum
from .numpy_functions import dot
from .numpy_functions import einsum
from .numpy_functions import linalg
