import unittest
import itertools
import numpy as np

import basis_array as basis
from test_array import TestCase, rand_orth_mat


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

MAXDIM = 4

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
        # 2D Hermition
        cls.dh_nn = dh_nn = np.random.rand(n, n)
        cls.dh_nn = dh_nn = (dh_nn + dh_nn.T)
        cls.ah_nn = basis.Array(dh_nn, basis=(bn, bn))
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

        cls.n2 = n2 = 11
        #cls.m2 = m2 = 12
        #cls.k2 = k2 = 13
        #cls.l2 = l2 = 14
        cls.bn2 = bn2 = basis.B(bn, rotation=rand_orth_mat(n, n2))

        cls.numpy_arrays_sq = [None, cls.d_n, cls.d_nn, cls.d_nnn, cls.d_nnnn]
        cls.basis_arrays_sq = [None, cls.a_n, cls.a_nn, cls.a_nnn, cls.a_nnnn]
        cls.numpy_arrays_rt = [None, cls.d_n, cls.d_nm, cls.d_nmk, cls.d_nmkl]
        cls.basis_arrays_rt = [None, cls.a_n, cls.a_nm, cls.a_nmk, cls.a_nmkl]


    def test_transpose_property(self):
        for ndim in range(1, 5):
            self.assertAllclose(self.basis_arrays_sq[ndim].T, self.numpy_arrays_sq[ndim].T)
            self.assertAllclose(self.basis_arrays_rt[ndim].T, self.numpy_arrays_rt[ndim].T)

    def test_transpose_square(self):
        for ndim in range(1, 5):
            for axes in itertools.permutations(range(ndim)):
                self.assertAllclose(self.basis_arrays_sq[ndim].transpose(axes), self.numpy_arrays_sq[ndim].transpose(axes))

    def test_transpose_rect(self):
        for ndim in range(1, 5):
            for axes in itertools.permutations(range(ndim)):
                self.assertAllclose(self.basis_arrays_rt[ndim].transpose(axes), self.numpy_arrays_rt[ndim].transpose(axes))

    def test_sum_square(self):
        for ndim in range(1, 5):
            for axis in powerset(range(ndim)):
                self.assertAllclose(self.basis_arrays_sq[ndim].sum(axis=axis),
                                    self.numpy_arrays_sq[ndim].sum(axis=axis))

    def test_sum_rect(self):
        for ndim in range(1, 5):
            for axis in powerset(range(ndim)):
                self.assertAllclose(self.basis_arrays_rt[ndim].sum(axis=axis),
                                    self.numpy_arrays_rt[ndim].sum(axis=axis))

    def test_trace_square(self):
        for ndim in range(2, 5):
            for axis1, axis2 in itertools.permutations(range(ndim), 2):
                self.assertAllclose(self.basis_arrays_sq[ndim].trace(axis1=axis1, axis2=axis2),
                                    self.numpy_arrays_sq[ndim].trace(axis1=axis1, axis2=axis2))

    def test_trace_rect(self):
        for ndim in range(2, 5):
            for axis1, axis2 in itertools.permutations(range(ndim), 2):
                with self.assertRaises(basis.util.BasisError):
                    self.assertAllclose(self.basis_arrays_rt[ndim].trace(axis1=axis1, axis2=axis2),
                                        self.numpy_arrays_rt[ndim].trace(axis1=axis1, axis2=axis2))

    def test_trace_subspace(self):
        tr1 = self.a_nn.as_basis((self.bn2, self.bn2)).trace()
        tr2 = self.a_nn.as_basis((self.bn2, self.bn2)).as_basis((self.bn, self.bn)).trace()
        self.assertAllclose(tr1, tr2, atol=1e-14, rtol=0)

    def test_eigh(self):
        # NumPy
        e, v = np.linalg.eigh(self.dh_nn)
        self.assertAllclose(np.einsum('ai,i,bi->ab', v, e, v), self.dh_nn)
        # Basis Array
        e, v = basis.linalg.eigh(self.ah_nn)
        self.assertAllclose(basis.einsum('ai,i,bi->ab', v, e, v), self.ah_nn)


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
