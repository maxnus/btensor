"""Defines BasisArray class, which allows the attachment of bases to NumPy arrays.
The arrays can then be used in a custom einsum function, where the overlaps between
the different bases are automatically taken into account.

Usage:

>>> root = Basis(mol.nao, metric=mf.get_ovlp())
>>> mo = Basis.add_basis(mf.mo_coeff, name='mo')
>>> mo_occ = Basis.add_basis(mf.mo_coeff[:,occ], name='mo-occ')
>>> mo_vir = Basis.add_basis(mf.mo_coeff[:,vir], name='mo-vir')
>>> fov = Tensor(fock[occ,vir], basis=(mo_occ, mo_vir))
>>> # View in different basis:
>>> print(fov.change_basis((mo, mo)))
>>> # Contract with other BasisArray:
>>> result = basis_einsum('ia,ja->ij', fov, t1)
>>> # The virtual dimension of `fov` and `t1` can be expressed in a different basis;
>>> # The corresponding overlap will be considered automatically.
>>> # `result` is another `BasisArray` instance.

"""

from .core import BasisInterface, Basis
from .core import nobasis
from .core import Tensor, Cotensor, Array, Coarray

from .tensorsum import TensorSum

from .numpy_functions import *
