import numpy as np

from btensor import TensorSum
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
