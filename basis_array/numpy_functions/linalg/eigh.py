import numpy as np
import basis_array
from basis_array.util.util import BasisError


def eigh(a):
    basis = a.basis[-1]
    if a.basis[-2].root != basis.root:
        raise BasisError
    e, v = np.linalg.eigh(a.value)
    eigenbasis = basis_array.Basis(v, parent=basis)
    cls = type(a)
    v = cls(v, basis=(a.basis[:-1] + (eigenbasis,)))
    e = cls(e, basis=eigenbasis)
    return e, v
