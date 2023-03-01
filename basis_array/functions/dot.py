from basis_array import Array
from basis_array.util import ndot, overlap


def dot(a, b):

    if a.ndim == b.ndim == 1:
        ovlp = overlap(a.basis[0], b.basis[0])
        return ndot(a.value, ovlp, b.value)

    if a.ndim == b.ndim == 2:
        ovlp = overlap(a.basis[-1], b.basis[0])
        out = ndot(a.value, ovlp, b.value)
        basis = (a.basis[0], b.basis[1])
    elif b.ndim == 1:
        ovlp = overlap(a.basis[-1], b.basis[0])
        out = ndot(a.value, ovlp, b.value)
        basis = a.basis[:-1]
    elif b.ndim >= 2:
        ovlp = overlap(a.basis[-1], b.basis[-2])
        out = ndot(a.value, ovlp, b.value)
        basis = (a.basis[:-1] + b.basis[:-2] + b.basis[-1:])

    return Array(out, basis=basis)
