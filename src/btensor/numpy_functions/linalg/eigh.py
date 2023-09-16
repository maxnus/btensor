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

import scipy

from btensor.util import BasisError, VarianceError
from btensor.basis import _Variance


def eigh(a):
    basis = a.basis[-1]
    variance = a.variance[-1]
    if a.basis[-2].root != basis.root:
        raise BasisError
    if a.variance[-2] != variance:
        raise VarianceError(f"variance needs to match between the last two axes")
    if basis.is_orthonormal:
        type_ = 1
        metric = None
    else:
        metric = basis.metric.to_numpy()
        type_ = 1 if (variance == _Variance.COVARIANT) else 2

    e, v = scipy.linalg.eigh(a.to_numpy(copy=False), b=metric, type=type_)
    eigenbasis = basis.make_subbasis(v, orthonormal=True)
    cls = type(a)
    v = cls(v, basis=(a.basis[:-1] + (eigenbasis,)), variance=(_Variance.CONTRAVARIANT, _Variance.COVARIANT))
    e = cls(e, basis=eigenbasis, variance=(_Variance.COVARIANT,))
    return e, v
