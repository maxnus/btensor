import unittest
import itertools
import numpy as np

import basis_array as basis
from test_array import TestCase, rand_orth_mat


class Tests(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.rand(0)
        cls.n = n = 21
        cls.m = m = 22
        cls.k = k = 23
        cls.l = l = 24
        cls.bn = bn = basis.B(n)
        cls.bm = bm = basis.B(m)
        cls.bk = bk = basis.B(k)
        cls.bl = bl = basis.B(l)

        # 1D
        cls.d_n = d_n = np.random.rand(n)
        cls.a_n = basis.Array(d_n, basis=bn)
        # 2D
        cls.d_nn = d_nn = np.random.rand(n, n)
        cls.d_nm = d_nm = np.random.rand(n, m)
        cls.a_nn = basis.Array(d_nn, basis=(bn, bn))
        cls.a_nm = basis.Array(d_nm, basis=(bn, bm))
        # 3D
        cls.d_nnn = d_nnn = np.random.rand(n, n, n)
        cls.d_nmk = d_nmk = np.random.rand(n, m, k)
        cls.a_nnn = basis.Array(d_nnn, basis=(bn, bn, bn))
        cls.a_nmk = basis.Array(d_nmk, basis=(bn, bm, bk))
        # 4D
        cls.d_nnnn = d_nnnn = np.random.rand(n, n, n, n)
        cls.d_nmkl = d_nmkl = np.random.rand(n, m, k, l)
        cls.a_nnnn = basis.Array(d_nnnn, basis=(bn, bn, bn, bn))
        cls.a_nmkl = basis.Array(d_nmkl, basis=(bn, bm, bk, bl))

        # --- Subspaces

        cls.ns = ns = 11

        cls.bns = bns = basis.B(bn, rand_orth_mat(n, ns))

    def test_transpose(self):
        # 1D
        self.assertAllclose(self.a_n.T, self.d_n.T)
        # 2D
        self.assertAllclose(self.a_nn.T, self.d_nn.T)
        self.assertAllclose(self.a_nm.T, self.d_nm.T)
        # 3D
        self.assertAllclose(self.a_nnn.T, self.d_nnn.T)
        self.assertAllclose(self.a_nmk.T, self.d_nmk.T)
        # 4D
        self.assertAllclose(self.a_nnnn.T, self.d_nnnn.T)
        self.assertAllclose(self.a_nmkl.T, self.d_nmkl.T)

    def test_transpose_axes(self):
        # 1D
        self.assertAllclose(self.a_n.transpose([0]), self.d_n.transpose([0]))
        # 2D
        for axes in itertools.permutations((1, 0)):
            self.assertAllclose(self.a_nn.transpose(axes), self.d_nn.transpose(axes))
            self.assertAllclose(self.a_nm.transpose(axes), self.d_nm.transpose(axes))
        # 3D
        for axes in itertools.permutations((0, 1, 2)):
            self.assertAllclose(self.a_nnn.transpose(axes), self.d_nnn.transpose(axes))
            self.assertAllclose(self.a_nmk.transpose(axes), self.d_nmk.transpose(axes))
        # 4D
        for axes in itertools.permutations((0, 1, 2, 3)):
            self.assertAllclose(self.a_nnnn.transpose(axes), self.d_nnnn.transpose(axes))
            self.assertAllclose(self.a_nmkl.transpose(axes), self.d_nmkl.transpose(axes))

    def test_sum(self):
        # 2D
        for axis in (None, 0, 1, (0, 1)):
            self.assertAllclose(self.a_nm.sum(axis=axis), self.d_nm.sum(axis=axis))
        # 3D
        self.assertAllclose(self.a_nmk.sum(), self.d_nmk.sum())
        for axis in (None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)):
            self.assertAllclose(self.a_nmk.sum(axis=axis), self.d_nmk.sum(axis=axis))


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
