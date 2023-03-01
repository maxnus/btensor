import unittest
import numpy as np

import basis_array as basis
from test_array import TestCase


class Tests(TestCase):

    def test_transpose(self):
        n, m = 30, 40
        a = np.random.rand(n, m)
        bn = basis.B(n)
        bm = basis.B(m)
        aa = basis.Array(a, basis=(bn, bm))
        self.assertAllclose(aa.T, a.T)


class DotTests(TestCase):

    def test_dot_11(self):
        n = 30
        a = np.random.rand(n)
        b = np.random.rand(n)
        c = np.dot(a, b)
        aa = basis.Array(a, basis=None)
        ab = basis.Array(b, basis=None)
        ac = basis.dot(aa, ab)
        self.assertAllclose(ac, c)

    def test_dot_21(self):
        n, m = 30, 40
        a = np.random.rand(n, m)
        b = np.random.rand(m)
        c = np.dot(a, b)
        aa = basis.Array(a, basis=(None, None))
        ab = basis.Array(b, basis=None)
        ac = basis.dot(aa, ab)
        self.assertAllclose(ac, c)

    def test_dot_31(self):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m, k)
        b = np.random.rand(k)
        c = np.dot(a, b)
        aa = basis.Array(a, basis=(None, None, None))
        ab = basis.Array(b, basis=None)
        ac = basis.dot(aa, ab)
        self.assertAllclose(ac, c)

    def test_dot_22(self):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m)
        b = np.random.rand(m, k)
        c = np.dot(a, b)
        aa = basis.Array(a, basis=(None, None))
        ab = basis.Array(b, basis=(None, None))
        ac = basis.dot(aa, ab)
        self.assertAllclose(ac, c)

    def test_dot_32(self):
        n, m, k, l = 30, 40, 50, 60
        a = np.random.rand(n, m, k)
        b = np.random.rand(k, l)
        c = np.dot(a, b)
        aa = basis.Array(a, basis=(None, None, None))
        ab = basis.Array(b, basis=(None, None))
        ac = basis.dot(aa, ab)
        self.assertAllclose(ac, c)


class EinsumTests(TestCase):

    def test_sum(self):
        n = 30
        a = np.random.rand(n)
        contract = 'i->'
        b = np.einsum(contract, a)
        bn = basis.B(n)
        aa = basis.Array(a, basis=bn)
        ab = basis.einsum(contract, aa)
        self.assertAllclose(ab, b)

    def test_trace(self):
        n = 30
        a = np.random.rand(n, n)
        contract = 'ii->'
        b = np.einsum(contract, a)
        bn = basis.B(n)
        aa = basis.Array(a, basis=(bn, bn))
        ab = basis.einsum(contract, aa)
        self.assertAllclose(ab, b)

    def test_matmul(self):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m)
        b = np.random.rand(m, k)
        contract = 'ij,jk->ik'
        c = np.einsum(contract, a, b)
        bn = basis.B(n)
        bm = basis.B(m)
        bk = basis.B(k)
        aa = basis.Array(a, basis=(bn, bm))
        ab = basis.Array(b, basis=(bm, bk))
        ac = basis.einsum(contract, aa, ab)
        self.assertAllclose(ac, c)

    def test_double_matmul(self):
        n, m, k, l = 30, 40, 50, 60
        a = np.random.rand(n, m)
        b = np.random.rand(m, k)
        c = np.random.rand(k, l)
        contract = 'ij,jk,kl->il'
        d = np.einsum(contract, a, b, c)
        bn = basis.B(n)
        bm = basis.B(m)
        bk = basis.B(k)
        bl = basis.B(l)
        aa = basis.Array(a, basis=(bn, bm))
        ab = basis.Array(b, basis=(bm, bk))
        ac = basis.Array(c, basis=(bk, bl))
        ad = basis.einsum(contract, aa, ab, ac)
        self.assertAllclose(ad, d)

    def test_trace_of_dot(self):
        n, m = 30, 40
        a = np.random.rand(n, m)
        b = np.random.rand(m, n)
        contract = 'ij,ji->'
        c = np.einsum(contract, a, b)
        bn = basis.B(n)
        bm = basis.B(m)
        aa = basis.Array(a, basis=(bn, bm))
        ab = basis.Array(b, basis=(bm, bn))
        ac = basis.einsum(contract, aa, ab)
        self.assertAllclose(ac, c)

    def test_ijk_kl_ijl(self):
        n, m, k, l = 30, 40, 50, 60
        a = np.random.rand(n, m, k)
        b = np.random.rand(k, l)
        contract = 'ijk,kl->ijl'
        c = np.einsum(contract, a, b)
        bn = basis.B(n)
        bm = basis.B(m)
        bk = basis.B(k)
        bl = basis.B(l)
        aa = basis.Array(a, basis=(bn, bm, bk))
        ab = basis.Array(b, basis=(bk, bl))
        ac = basis.einsum(contract, aa, ab)
        self.assertAllclose(ac, c)




if __name__ == '__main__':
    unittest.main()
