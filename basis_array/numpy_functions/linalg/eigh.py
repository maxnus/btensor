import numpy as np
import basis_array
from basis_array.util import BasisError


def eigh(a):

    basis = a.basis[-1]
    if a.basis[-2].root != basis.root:
        raise BasisError

    e, v = np.linalg.eigh(a.value)

    eigenbasis = basis_array.B(basis, rotation=v)

    cls = type(a)
    v = cls(v, basis=(a.basis[:-1] +  (eigenbasis,)))
    e = cls(e, basis=eigenbasis)

    return e, v
