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

__version__ = '0.0.0'

#     Copyright 2023 Max Nusspickel
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import sys

from loguru import logger
#logger.disable('btensor')
#logger.configure(
#    handlers=[
#        dict(sink=sys.stderr, format="[{time}] {message}", colors=True),
#        dict(sink='btensor.log', format="[{time}] {message}")
#    ]
#)

from .core import BasisInterface, Basis
from .core import nobasis
from .core import Tensor, Cotensor, Array, Coarray

from .tensorsum import TensorSum

from .numpy_functions import *
