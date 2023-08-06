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

import pytest
import numpy as np

import btensor as bt
from helper import TestCase


@pytest.fixture(params=['with-shape', ''], scope='module')
def with_shape(request, shape_and_basis):
    if request.param:
        return shape_and_basis[0]
    return None


class TestNumpyFunctions(TestCase):

    @pytest.mark.parametrize('funcs', [(bt.empty, np.empty), (bt.zeros, np.zeros), (bt.ones, np.ones)],
                             ids=['empty', 'zeros', 'ones'])
    def test_empty_zeros_ones(self, funcs, shape_and_basis, with_shape):
        bt_func, np_func = funcs
        shape, basis = shape_and_basis
        expected = np_func(shape)
        if with_shape is None and bt.nobasis in basis:
            with pytest.raises(ValueError):
                bt_func(basis)
            return
        tensor = bt_func(basis, shape=with_shape)
        assert tensor.shape == expected.shape
        assert tensor.dtype == expected.dtype
        if np_func is np.empty:
            return
        self.assert_allclose(tensor, expected)

    @pytest.mark.parametrize('funcs', [(bt.empty_like, np.empty_like), (bt.zeros_like, np.zeros_like),
                                       (bt.ones_like, np.ones_like)], ids=['empty', 'zeros', 'ones'])
    def test_empty_zeros_ones_like(self, funcs, shape_and_basis):
        bt_func, np_func = funcs
        shape, basis = shape_and_basis
        values = np.random.random(shape)
        tensor = bt.Tensor(values, basis=basis)
        expected = np_func(values)
        assert bt_func(expected).shape == expected.shape
        assert bt_func(expected).dtype == expected.dtype
        assert bt_func(tensor).shape == expected.shape
        assert bt_func(tensor).dtype == expected.dtype
        if np_func is np.empty_like:
            return
        self.assert_allclose(bt_func(expected), expected)
        self.assert_allclose(bt_func(tensor), expected)

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
        tr1 = array.project((subbasis, subbasis)).trace()
        tr2 = array.project((subbasis, subbasis)).change_basis(array.basis).trace()
        self.assert_allclose(tr1, tr2, atol=1e-14, rtol=0)

    def test_eigh(self, get_array):
        array, np_array = get_array(ndim=2, hermitian=True)
        eig_expected = np.linalg.eigh(np_array)[0]
        eig, eigv = bt.linalg.eigh(array)
        self.assert_allclose(eig, eig_expected)
        self.assert_allclose(bt.einsum('ai,i,bi->ab', eigv, eig, eigv), np_array)


class TestDot(TestCase):

    def test_dot_11(self, tensor_cls_2x):
        n = 30
        a = np.random.rand(n)
        b = np.random.rand(n)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=bt.nobasis)
        ab = tensor_cls_2x[1](b, basis=bt.nobasis)
        ac = bt.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_21(self, tensor_cls_2x):
        n, m = 30, 40
        a = np.random.rand(n, m)
        b = np.random.rand(m)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(bt.nobasis, bt.nobasis))
        ab = tensor_cls_2x[1](b, basis=bt.nobasis)
        ac = bt.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_31(self, tensor_cls_2x):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m, k)
        b = np.random.rand(k)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(bt.nobasis, bt.nobasis, bt.nobasis))
        ab = tensor_cls_2x[1](b, basis=bt.nobasis)
        ac = bt.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_22(self, tensor_cls_2x):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m)
        b = np.random.rand(m, k)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(bt.nobasis, bt.nobasis))
        ab = tensor_cls_2x[1](b, basis=(bt.nobasis, bt.nobasis))
        ac = bt.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_32(self, tensor_cls_2x):
        n, m, k, l = 30, 40, 50, 60
        a = np.random.rand(n, m, k)
        b = np.random.rand(k, l)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(bt.nobasis, bt.nobasis, bt.nobasis))
        ab = tensor_cls_2x[1](b, basis=(bt.nobasis, bt.nobasis))
        ac = bt.dot(aa, ab)
        self.assert_allclose(ac, c)
