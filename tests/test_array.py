import numpy as np
from test_tensor import TestTensor
from test_tensor import TestArithmetricSameBasis
from test_tensor import TestArithmetricDifferentBasis
from test_tensor import TestSubbasis
from btensor import Array


class TestArray(TestTensor):

    tensor_cls = Array


class TestArithmetricSameBasisArray(TestArithmetricSameBasis):

    tensor_cls = Array

    def test_getitem_int(self):
        self.assert_allclose(self.a1[0]._data, self.d1[0])
        self.assert_allclose(self.a1[-1]._data, self.d1[-1])

    def test_getitem_slice(self):
        self.assert_allclose(self.a1[:2]._data, self.d1[:2])
        self.assert_allclose(self.a1[::2]._data, self.d1[::2])
        self.assert_allclose(self.a1[:-1]._data, self.d1[:-1])

    def test_getitem_elipsis(self):
        self.assert_allclose(self.a1[...]._data, self.d1[...])

    def test_getitem_newaxis(self):
        self.assert_allclose(self.a1[np.newaxis], self.d1[np.newaxis])

    def test_getitem_list_array(self):
        self.assert_allclose(self.a1[[0]], self.d1[[0]])
        self.assert_allclose(self.a1[np.asarray([0])], self.d1[np.asarray([0])])
        self.assert_allclose(self.a1[[0, 2, 1]], self.d1[[0, 2, 1]])
        self.assert_allclose(self.a1[np.asarray([0, 2, 1])], self.d1[np.asarray([0, 2, 1])])
        self.assert_allclose(self.a1[[-2, 1]], self.d1[[-2, 1]])
        self.assert_allclose(self.a1[np.asarray([-2, 1])], self.d1[np.asarray([-2, 1])])

    def test_getitem_tuple(self):
        self.assert_allclose(self.a1[0, 2], self.d1[0, 2])
        self.assert_allclose(self.a1[-1, 2], self.d1[-1, 2])
        self.assert_allclose(self.a1[:2, 3:]._data, self.d1[:2, 3:])
        self.assert_allclose(self.a1[::2, ::-1]._data, self.d1[::2, ::-1])
        self.assert_allclose(self.a1[0, :2]._data, self.d1[0, :2])
        self.assert_allclose(self.a1[::-1, -1]._data, self.d1[::-1, -1])
        self.assert_allclose(self.a1[..., 0], self.d1[..., 0])
        self.assert_allclose(self.a1[..., :2], self.d1[..., :2])
        self.assert_allclose(self.a1[::-1, ...], self.d1[::-1, ...])
        self.assert_allclose(self.a1[np.newaxis, 0], self.d1[np.newaxis, 0])
        self.assert_allclose(self.a1[:2, np.newaxis], self.d1[:2, np.newaxis])

    #def test_getitem_boolean(self):
    #    mask = [True, True, False, False, True]
    #    self.assertAllclose(self.a1[mask].value, self.d1[mask])

    def test_setitem(self):
        def test(key, value):
            a1 = self.a1.copy()
            d1 = self.d1.copy()
            a1[key] = value
            d1[key] = value
            self.assert_allclose(a1, d1)

        test(0, 0)
        test(slice(None), 0)
        test(slice(1, 2), 0)
        test(..., 0)
        test([1, 2], 0)
        test(([1, 2, -1], [0, 3, 2]), 0)
        test((0, 2), 0)
        test((slice(None), 2), 0)
        test((slice(1, 2), slice(3, 0, -1)), 0)
        test((slice(1, 2), [0, 2]), 0)
        test(np.newaxis, 0)


class TestArithmetricDifferentBasisArray(TestArithmetricDifferentBasis):

    tensor_cls = Array


class TestSubbasisArray(TestSubbasis):

    tensor_cls = Array
