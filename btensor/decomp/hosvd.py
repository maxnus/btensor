from __future__ import annotations
import typing

import numpy as np
import scipy
import scipy.linalg

from btensor import Basis
if typing.TYPE_CHECKING:
    from btensor import Tensor


def hosvd(tensor: Tensor, svtol: float | None = None):
    """Calculate core tensor of higher order SVD (HOSVD)."""
    if tensor.ndim < 3:
        raise NotImplementedError(f"cannot perform HOSVD for {tensor.ndim}-dimensional tensor.")
    array = tensor.to_numpy(copy=False)
    core = array
    basis = []
    for dim, bas in enumerate(tensor.basis):
        if not isinstance(bas, Basis):
            raise RuntimeError
        a = np.moveaxis(array, dim, 0).reshape((array.shape[dim], -1))
        u, s, _ = scipy.linalg.svd(a, check_finite=False)
        if svtol is not None:
            u = u[:, s >= svtol]
        basis.append(bas.make_subbasis(u))
        core = np.tensordot(core, u.T.conj(), axes=(0, 1))
    return type(tensor)(core, basis=tuple(basis))
