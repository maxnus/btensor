import pytest
import itertools
import string
import numpy as np

import btensor as bt
from helper import TestCase


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


class TestEinsum(TestCase):

    def test_summation(self, einsum_summation, get_array):
        ndim = len(einsum_summation.split('->')[0])
        array, data = get_array(ndim)
        self.assert_allclose(bt.einsum(einsum_summation, array), np.einsum(einsum_summation, data))

    def test_contraction(self, einsum_contraction, get_array):
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