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
from typing import *

import pytest
import numpy as np

from btensor import Tensor

if TYPE_CHECKING:
    from btensor import Basis
    from btensor.basis import NBasis


class TestTensor:

    def __init__(self,
                 array: np.ndarray, basis: Basis | Tuple[Basis, ...],
                 variance: Tuple[int, ...] | None = None,
                 numpy_compatible: bool = True) -> None:
        self.array = array
        self.basis = basis
        self.variance = variance
        self.numpy_compatible = numpy_compatible
        self.tensor = Tensor(array, basis=basis, variance=variance, numpy_compatible=numpy_compatible)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape


@pytest.fixture(scope='module')
def get_test_tensor():
    def tensor_factory(basis: NBasis,
                       number: int = 1,
                       hermitian: bool = False,
                       numpy_compatible: bool = True) -> TestTensor | List[TestTensor]:
        np.random.seed(0)
        result = []
        for n in range(number):
            data = np.random.random(tuple([b.size for b in basis]))
            if hermitian:
                data = (data + data.T)/2
            result.append(TestTensor(data, basis=basis, numpy_compatible=numpy_compatible))
        if number == 1:
            return result[0]
        return result
    return tensor_factory


@pytest.fixture(params=[True, False], ids=['npc', 'nnpc'], scope='module')
def numpy_compatible(request):
    return request.param


@pytest.fixture(scope='module')
def test_tensor(get_test_tensor, basis_large, ndim, numpy_compatible):
    return get_test_tensor(basis=ndim * (basis_large,), numpy_compatible=numpy_compatible)


@pytest.fixture(scope='module')
def test_tensor_2(get_test_tensor, basis_large, ndim, numpy_compatible):
    return get_test_tensor(basis=ndim * (basis_large,), numpy_compatible=numpy_compatible)
