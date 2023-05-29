import pytest
import itertools
import string
import numpy as np

import btensor as bt
from helper import TestCase
from conftest import get_permutations_of_combinations


def loop_einsum_subscripts(ndim, start_label=0):
    indices = list(string.ascii_lowercase)[start_label:start_label+ndim]
    for nsum in range(0, ndim+1):
        for sumindices in itertools.combinations(range(ndim), nsum):
            subscripts = indices.copy()
            for sumidx in sumindices:
                subscripts[sumidx] = 'X'
            subscripts = ''.join(subscripts)
            yield subscripts


def generate_einsum_summation(maxdim):
    for ndim in range(1, maxdim+1):
        for sub in loop_einsum_subscripts(ndim):
            for include_result in [True, False]:
                if include_result:
                    summation = sub + '->' + sub.replace('X', '')
                else:
                    summation = sub
                yield summation


def generate_einsum_contraction(maxdim):
    for ndim1 in range(1, maxdim + 1):
        for sub1 in loop_einsum_subscripts(ndim1):
            for ndim2 in range(1, maxdim + 1):
                for sub2 in loop_einsum_subscripts(ndim2, start_label=ndim1):
                    sub = ','.join([sub1, sub2])
                    for include_result in [True, False]:
                        if include_result:
                            contraction = sub + '->' + (sub1 + sub2).replace('X', '')
                        else:
                            contraction = sub
                        yield contraction


@pytest.fixture(params=generate_einsum_summation(maxdim=4))
def einsum_summation(request):
    return request.param


@pytest.fixture(params=generate_einsum_contraction(maxdim=4))
def einsum_contraction(request):
    return request.param


class TestNumpyFunctions(TestCase):

    #@classmethod
    #def setup_class(cls):
    #    np.random.rand(0)
    #    cls.n = n = 5
    #    cls.m = m = 6
    #    cls.k = k = 7
    #    cls.l = l = 8
    #    cls.bn = bn = bt.Basis(n)
    #    cls.bm = bm = bt.Basis(m)
    #    cls.bk = bk = bt.Basis(k)
    #    cls.bl = bl = bt.Basis(l)
    #    # 1D
    #    cls.d_n = d_n = np.random.rand(n)
    #    cls.a_n = cls.tensor_cls(d_n, basis=bn)
    #    # 2D
    #    cls.d_nn = d_nn = np.random.rand(n, n)
    #    cls.d_nm = d_nm = np.random.rand(n, m)
    #    cls.a_nn = cls.tensor_cls(d_nn, basis=(bn, bn))
    #    cls.a_nm = cls.tensor_cls(d_nm, basis=(bn, bm))
    #    # 2D Hermitian
    #    cls.dh_nn = dh_nn = np.random.rand(n, n)
    #    cls.dh_nn = dh_nn = (dh_nn + dh_nn.T)
    #    cls.ah_nn = cls.tensor_cls(dh_nn, basis=(bn, bn))
    #    # 3D
    #    cls.d_nnn = d_nnn = np.random.rand(n, n, n)
    #    cls.d_nmk = d_nmk = np.random.rand(n, m, k)
    #    cls.a_nnn = cls.tensor_cls(d_nnn, basis=(bn, bn, bn))
    #    cls.a_nmk = cls.tensor_cls(d_nmk, basis=(bn, bm, bk))
    #    # 4D
    #    cls.d_nnnn = d_nnnn = np.random.rand(n, n, n, n)
    #    cls.d_nmkl = d_nmkl = np.random.rand(n, m, k, l)
    #    cls.a_nnnn = cls.tensor_cls(d_nnnn, basis=(bn, bn, bn, bn))
    #    cls.a_nmkl = cls.tensor_cls(d_nmkl, basis=(bn, bm, bk, bl))

    #    # --- Subspaces

    #    cls.n2 = n2 = 3
    #    #cls.m2 = m2 = 12
    #    #cls.k2 = k2 = 13
    #    #cls.l2 = l2 = 14
    #    cls.bn2 = bn2 = bt.Basis(rand_orth_mat(n, n2), parent=bn)

    #    cls.numpy_arrays_sq = [None, cls.d_n, cls.d_nn, cls.d_nnn, cls.d_nnnn]
    #    cls.basis_arrays_sq = [None, cls.a_n, cls.a_nn, cls.a_nnn, cls.a_nnnn]
    #    cls.numpy_arrays_rt = [None, cls.d_n, cls.d_nm, cls.d_nmk, cls.d_nmkl]
    #    cls.basis_arrays_rt = [None, cls.a_n, cls.a_nm, cls.a_nmk, cls.a_nmkl]


    # NumPy

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

    @pytest.mark.parametrize('key', [..., (0, ...), (..., 0), (0, 0, ...), (0, ..., 0), (..., 0, 0),
                                     (slice(None), ...), (..., slice(None)), (slice(None), slice(None), ...),
                                     (slice(None), ..., slice(None)), (..., slice(None), slice(None)),
                                     (0, slice(None), ...), (0, ..., slice(None)), (..., 0, slice(None)),
                                     (slice(None), 0, ...), (slice(None), ..., 0), (..., slice(None), 0)],
                             ids=lambda x: str(x).replace('slice(None, None, None)', ':').replace('Ellipsis', '...'))
    def test_getitem_with_ellipsis(self, ndim_atleast2, get_array, key):
        array, np_array = get_array(ndim=ndim_atleast2)
        self.assert_allclose(array[key], np_array[key])

    def test_newaxis(self):
        assert bt.newaxis == np.newaxis

    @pytest.mark.parametrize('key', get_permutations_of_combinations([bt.newaxis, slice(None)], 2),
                             ids=lambda x: str(x))
    def test_getitem_newaxis(self, ndim_atleast2, get_array, key):
        array, np_array = get_array(ndim_atleast2)
        self.assert_allclose(array[key], np_array[key])

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


class TestEinsum(TestCase):

    def test_einsum_summation(self, einsum_summation, get_array):
        ndim = len(einsum_summation.split('->')[0])
        array, data = get_array(ndim)
        self.assert_allclose(bt.einsum(einsum_summation, array), np.einsum(einsum_summation, data))

    def test_einsum_contraction(self, einsum_contraction, get_array):
        ndim1, ndim2 = [len(x) for x in einsum_contraction.split('->')[0].split(',')]
        array1, data1 = get_array(ndim1)
        array2, data2 = get_array(ndim2)
        self.assert_allclose(bt.einsum(einsum_contraction, array1, array2), np.einsum(einsum_contraction, data1, data2))

    def test_matmul(self, tensor_cls_2x):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m)
        b = np.random.rand(m, k)
        contract = 'ij,jk->ik'
        c = np.einsum(contract, a, b)
        bn = bt.Basis(n)
        bm = bt.Basis(m)
        bk = bt.Basis(k)
        aa = tensor_cls_2x[0](a, basis=(bn, bm))
        ab = tensor_cls_2x[1](b, basis=(bm, bk))
        ac = bt.einsum(contract, aa, ab)
        self.assert_allclose(ac, c)

    def test_double_matmul(self, tensor_cls_3x):
        n, m, k, l = 30, 40, 50, 60
        a = np.random.rand(n, m)
        b = np.random.rand(m, k)
        c = np.random.rand(k, l)
        contract = 'ij,jk,kl->il'
        d = np.einsum(contract, a, b, c)
        bn = bt.Basis(n)
        bm = bt.Basis(m)
        bk = bt.Basis(k)
        bl = bt.Basis(l)
        aa = tensor_cls_3x[0](a, basis=(bn, bm))
        ab = tensor_cls_3x[1](b, basis=(bm, bk))
        ac = tensor_cls_3x[2](c, basis=(bk, bl))
        ad = bt.einsum(contract, aa, ab, ac)
        self.assert_allclose(ad, d)

    def test_trace_of_dot(self, tensor_cls_2x):
        n, m = 30, 40
        a = np.random.rand(n, m)
        b = np.random.rand(m, n)
        contract = 'ij,ji->'
        c = np.einsum(contract, a, b)
        bn = bt.Basis(n)
        bm = bt.Basis(m)
        aa = tensor_cls_2x[0](a, basis=(bn, bm))
        ab = tensor_cls_2x[1](b, basis=(bm, bn))
        ac = bt.einsum(contract, aa, ab)
        self.assert_allclose(ac, c)

    def test_ijk_kl_ijl(self, tensor_cls_2x):
        n, m, k, l = 30, 40, 50, 60
        a = np.random.rand(n, m, k)
        b = np.random.rand(k, l)
        contract = 'ijk,kl->ijl'
        c = np.einsum(contract, a, b)
        bn = bt.Basis(n)
        bm = bt.Basis(m)
        bk = bt.Basis(k)
        bl = bt.Basis(l)
        aa = tensor_cls_2x[0](a, basis=(bn, bm, bk))
        ab = tensor_cls_2x[1](b, basis=(bk, bl))
        ac = bt.einsum(contract, aa, ab)
        self.assert_allclose(ac, c)
