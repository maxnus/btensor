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
import itertools

import pytest
import numpy as np

import btensor
from helper import TestCase


@pytest.fixture(params=['with-shape', ''], scope='module')
def with_shape(request, shape_and_basis):
    if request.param:
        return shape_and_basis[0]
    return None


class TestNumpyFunctions(TestCase):

    @pytest.mark.parametrize('funcs', [(btensor.empty, np.empty), (btensor.zeros, np.zeros), (btensor.ones, np.ones)],
                             ids=['empty', 'zeros', 'ones'])
    def test_empty_zeros_ones(self, funcs, shape_and_basis, with_shape):
        bt_func, np_func = funcs
        shape, basis = shape_and_basis
        expected = np_func(shape)
        if with_shape is None and btensor.nobasis in basis:
            with pytest.raises(ValueError):
                bt_func(basis)
            return
        tensor = bt_func(basis, shape=with_shape)
        assert tensor.current_shape == expected.shape
        assert tensor.dtype == expected.dtype
        if np_func is np.empty:
            return
        self.assert_allclose(tensor, expected)

    @pytest.mark.parametrize('funcs', [(btensor.empty_like, np.empty_like), (btensor.zeros_like, np.zeros_like),
                                       (btensor.ones_like, np.ones_like)], ids=['empty', 'zeros', 'ones'])
    def test_empty_zeros_ones_like(self, funcs, shape_and_basis):
        bt_func, np_func = funcs
        shape, basis = shape_and_basis
        values = np.random.random(shape)
        tensor = btensor.Tensor(values, basis=basis)
        expected = np_func(values)
        assert bt_func(expected).current_shape == expected.shape
        assert bt_func(expected).dtype == expected.dtype
        assert bt_func(tensor).current_shape == expected.shape
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
        self.assert_allclose(btensor.sum(array), expected)
        self.assert_allclose(btensor.sum(np_array), expected)

    def test_sum_with_axis(self, ndim_and_axis, get_array):
        ndim, axis = ndim_and_axis
        array, np_array = get_array(ndim)
        self.assert_allclose(array.sum(axis=axis), np_array.sum(axis=axis))

    def test_trace(self, ndim_atleast2, get_array):
        array, np_array = get_array(ndim_atleast2)
        self.assert_allclose(array.trace(), np_array.trace())

    def test_trace_nonorthogonal(self, rng):
        n = 10
        basis_orth = btensor.Basis(n)
        basis_nonorth = basis_orth.make_subbasis(rng.random((n, n)))
        tensor = btensor.Tensor(rng.random((n, n)), basis=(basis_orth, basis_orth))
        expected = tensor.trace()
        result = tensor[basis_nonorth, basis_nonorth].trace()
        self.assert_allclose(expected, result, atol=1e-12, rtol=0)

    def test_einsum_trace_nonorthogonal(self, rng):
        n = 10
        basis_orth = btensor.Basis(n)
        basis_nonorth = basis_orth.make_subbasis(rng.random((n, n)))
        tensor = btensor.Tensor(rng.random((n, n)), basis=(basis_orth, basis_orth))
        expected = tensor.trace()
        result = btensor.einsum('ii->', tensor[basis_nonorth, basis_nonorth])
        self.assert_allclose(expected, result, atol=1e-12, rtol=0)

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
        eig, eigv = btensor.linalg.eigh(array)
        self.assert_allclose(eig, eig_expected)
        eigmat = btensor.Tensor(np.diag(eig.to_numpy()), basis=2*eig.basis)
        self.assert_allclose(btensor.einsum('ai,ij,bj->ab', eigv, eigmat, eigv), np_array)

    @pytest.mark.parametrize('dest', list(itertools.permutations([0, 1, 2])), ids=str)
    def test_moveaxis_3d(self, dest, get_array, ndim_atleast2):
        array, np_array = get_array(ndim=3)
        # Remove a bit along the first axis, to make array non-square:
        b = array.basis[1].make_subbasis(slice(0, 2))
        array = array[:, b]
        np_array = np_array[:, :2]
        source = (0, 1, 2)
        expected = np.moveaxis(np_array, source, dest)
        result = btensor.moveaxis(array, source, dest)
        self.assert_allclose(result.to_numpy(), expected)


class TestDot(TestCase):

    def test_dot_11(self, tensor_cls_2x):
        n = 30
        a = np.random.rand(n)
        b = np.random.rand(n)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=btensor.nobasis)
        ab = tensor_cls_2x[1](b, basis=btensor.nobasis)
        ac = btensor.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_21(self, tensor_cls_2x):
        n, m = 30, 40
        a = np.random.rand(n, m)
        b = np.random.rand(m)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(btensor.nobasis, btensor.nobasis))
        ab = tensor_cls_2x[1](b, basis=btensor.nobasis)
        ac = btensor.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_31(self, tensor_cls_2x):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m, k)
        b = np.random.rand(k)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(btensor.nobasis, btensor.nobasis, btensor.nobasis))
        ab = tensor_cls_2x[1](b, basis=btensor.nobasis)
        ac = btensor.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_22(self, tensor_cls_2x):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m)
        b = np.random.rand(m, k)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(btensor.nobasis, btensor.nobasis))
        ab = tensor_cls_2x[1](b, basis=(btensor.nobasis, btensor.nobasis))
        ac = btensor.dot(aa, ab)
        self.assert_allclose(ac, c)

    def test_dot_32(self, tensor_cls_2x):
        n, m, k, l = 30, 40, 50, 60
        a = np.random.rand(n, m, k)
        b = np.random.rand(k, l)
        c = np.dot(a, b)
        aa = tensor_cls_2x[0](a, basis=(btensor.nobasis, btensor.nobasis, btensor.nobasis))
        ab = tensor_cls_2x[1](b, basis=(btensor.nobasis, btensor.nobasis))
        ac = btensor.dot(aa, ab)
        self.assert_allclose(ac, c)
