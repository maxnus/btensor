import numpy as np
import btensor
from btensor.util.util import BasisError


def eigh(a):
    basis = a.basis[-1]
    if a.basis[-2].root != basis.root:
        raise BasisError
    e, v = np.linalg.eigh(a._value)
    eigenbasis = btensor.Basis(v, parent=basis)
    cls = type(a)
    v = cls(v, basis=(a.basis[:-1] + (eigenbasis,)))
    e = cls(e, basis=eigenbasis)
    return e, v
