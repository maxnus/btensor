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

import numpy as np

from btensor import TensorSum, einsum
from helper import TestCase


class TestTensorsum(TestCase):

    def test_tensorsum(self, get_tensor):
        tensor, np_array = get_tensor(ndim=2)
        tensor2 = tensor.copy()
        tensor_sum = TensorSum([tensor, tensor2])
        self.assert_allclose(tensor_sum.evaluate(), 2*np_array)

    def test_tensorsum2(self, get_tensor):
        tensor, np_array = get_tensor(ndim=2)
        tensor2 = tensor.copy()
        tensor_sum = TensorSum([tensor, tensor2])
        lhs = tensor_sum.dot(tensor_sum).evaluate()
        self.assert_allclose(lhs, 4*np.dot(np_array, np_array))

    def test_tensorsum_2x1(self, get_tensor):
        tensors, arrays = zip(*get_tensor(ndim=2, number=3))
        ts1 = TensorSum(tensors[:2])
        t2 = tensors[2]
        subscripts = 'ij,jk->ik'
        expected = np.einsum(subscripts, ts1.evaluate().to_numpy(), t2.to_numpy())
        result = einsum('ij,jk->ik', ts1, t2).to_numpy()
        self.assert_allclose(result, expected)

    def test_tensorsum_2x2(self, get_tensor):
        tensors, arrays = zip(*get_tensor(ndim=2, number=4))
        for i, t in enumerate(tensors):
            t.name = f'Tensor{i}'
        ts1 = TensorSum(tensors[:2])
        ts2 = TensorSum(tensors[2:])
        subscripts = 'ij,jk->ik'
        expected = np.einsum(subscripts, ts1.to_numpy(), ts2.to_numpy())
        result = einsum('ij,jk->ik', ts1, ts2).to_numpy()
        self.assert_allclose(result, expected)
