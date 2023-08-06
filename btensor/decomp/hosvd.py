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
