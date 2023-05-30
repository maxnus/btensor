import pytest
import numpy as np

import btensor as bt
from helper import TestCase


class TestNumpyFunctions(TestCase):

    def test_empty(self, shape, basis_for_shape):
        expected = np.empty(shape)
        tensor = bt.empty(basis_for_shape)
        assert tensor.shape == expected.shape
        assert tensor.dtype == expected.dtype

    def test_empty_like(self, shape, basis_for_shape):
        expected = np.empty(shape)
        tensor = bt.empty(basis_for_shape)
        assert bt.empty_like(expected).shape == expected.shape
        assert bt.empty_like(expected).dtype == expected.dtype
        assert bt.empty_like(tensor).shape == expected.shape
        assert bt.empty_like(tensor).dtype == expected.dtype

    def test_ones(self, shape, basis_for_shape):
        expected = np.ones(shape)
        tensor = bt.ones(basis_for_shape)
        self.assert_allclose(tensor, expected)

    def test_ones_like(self, shape, basis_for_shape):
        expected = np.ones(shape)
        tensor = bt.ones(basis_for_shape)
        self.assert_allclose(bt.ones_like(expected), expected)
        self.assert_allclose(bt.ones_like(tensor), expected)

    def test_zeros(self, shape, basis_for_shape):
        expected = np.zeros(shape)
        tensor = bt.zeros(basis_for_shape)
        self.assert_allclose(tensor, expected)

    def test_zeros_like(self, shape, basis_for_shape):
        expected = np.zeros(shape)
        tensor = bt.zeros(basis_for_shape)
        self.assert_allclose(bt.zeros_like(expected), expected)
        self.assert_allclose(bt.zeros_like(tensor), expected)

    def test_transpose_property(self, tensor_or_array):
        tensor, np_array = tensor_or_array
        self.assert_allclose(tensor.T, np_array.T)

    def test_transpose_square(self, ndim_and_same_size_axis, get_tensor_or_array, tensor_cls):
        ndim, axis = ndim_and_same_size_axis
        tensor, np_array = get_tensor_or_array(ndim=ndim, tensor_cls=tensor_cls)
        self.assert_allclose(tensor.transpose(axis), np_array.transpose(axis))

    def test_sum(self, array):
        array, np_array = array
        expected = np.sum(np_array)
        self.assert_allclose(array.sum(), expected)
        self.assert_allclose(bt.sum(array), expected)
        self.assert_allclose(bt.sum(np_array), expected)

    def test_sum_with_axis(self, ndim_and_axis, get_array):
        ndim, axis = ndim_and_axis
        array, np_array = get_array(ndim)
        self.assert_allclose(array.sum(axis=axis), np_array.sum(axis=axis))

    def test_trace(self, ndim_atleast2, get_array):
        array, np_array = get_array(ndim_atleast2)
        self.assert_allclose(array.trace(), np_array.trace())

    def test_trace_with_axis(self, ndim_axis1_axis2, get_array):
        ndim, axis1, axis2 = ndim_axis1_axis2
        array, np_array = get_array(ndim)
        self.assert_allclose(array.trace(axis1=axis1, axis2=axis2), np_array.trace(axis1=axis1, axis2=axis2))

    def test_trace_subspace(self, ndim_atleast2, subbasis, get_array):
        array, np_array = get_array(ndim_atleast2)
        tr1 = array.proj((subbasis, subbasis)).trace()
        tr2 = array.proj((subbasis, subbasis)).as_basis(array.basis).trace()
        self.assert_allclose(tr1, tr2, atol=1e-14, rtol=0)

    def test_newaxis(self):
        assert bt.newaxis == np.newaxis

    def test_eigh(self, get_array):
        array, np_array = get_array(ndim=2, hermitian=True)
        eig_expected = np.linalg.eigh(np_array)[0]
        eig, eigv = bt.linalg.eigh(array)
        self.assert_allclose(eig, eig_expected)
        self.assert_allclose(bt.einsum('ai,i,bi->ab', eigv, eig, eigv), np_array)


nobasis = -1


class TestDot(TestCase):

    def test_dot_11(self, tensor_cls_2x):
        n = 30
        a = np.random.rand(n)
        b = np.random.rand(n)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=nobasis)
        ab = tensor_cls_2x[1](b, basis=nobasis)
        ac = bt.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_21(self, tensor_cls_2x):
        n, m = 30, 40
        a = np.random.rand(n, m)
        b = np.random.rand(m)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(nobasis, nobasis))
        ab = tensor_cls_2x[1](b, basis=nobasis)
        ac = bt.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_31(self, tensor_cls_2x):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m, k)
        b = np.random.rand(k)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(nobasis, nobasis, nobasis))
        ab = tensor_cls_2x[1](b, basis=nobasis)
        ac = bt.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_22(self, tensor_cls_2x):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m)
        b = np.random.rand(m, k)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(nobasis, nobasis))
        ab = tensor_cls_2x[1](b, basis=(nobasis, nobasis))
        ac = bt.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_32(self, tensor_cls_2x):
        n, m, k, l = 30, 40, 50, 60
        a = np.random.rand(n, m, k)
        b = np.random.rand(k, l)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(nobasis, nobasis, nobasis))
        ab = tensor_cls_2x[1](b, basis=(nobasis, nobasis))
        ac = bt.dot(aa, ab)
        self.assert_allclose(ac, c)
