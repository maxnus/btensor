import unittest
import itertools
import functools
import string
import numpy as np

import basis_array as basis
from test_array import TestCase, rand_orth_mat


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


class Tests(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.rand(0)
        cls.n = n = 5
        cls.m = m = 6
        cls.k = k = 7
        cls.l = l = 8
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
        # 2D Hermitian
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

        cls.n2 = n2 = 3
        #cls.m2 = m2 = 12
        #cls.k2 = k2 = 13
        #cls.l2 = l2 = 14
        cls.bn2 = bn2 = basis.B(bn, rotation=rand_orth_mat(n, n2))

        cls.numpy_arrays_sq = [None, cls.d_n, cls.d_nn, cls.d_nnn, cls.d_nnnn]
        cls.basis_arrays_sq = [None, cls.a_n, cls.a_nn, cls.a_nnn, cls.a_nnnn]
        cls.numpy_arrays_rt = [None, cls.d_n, cls.d_nm, cls.d_nmk, cls.d_nmkl]
        cls.basis_arrays_rt = [None, cls.a_n, cls.a_nm, cls.a_nmk, cls.a_nmkl]

    # NumPy

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
        tr1 = self.a_nn.project_onto((self.bn2, self.bn2)).trace()
        tr2 = self.a_nn.project_onto((self.bn2, self.bn2)).as_basis((self.bn, self.bn)).trace()
        self.assertAllclose(tr1, tr2, atol=1e-14, rtol=0)

    def test_getitem_with_ellipsis(self):
        for ndim in range(2, 5):
            self.assertAllclose(self.basis_arrays_rt[ndim][...],
                                self.numpy_arrays_rt[ndim][...])
            self.assertAllclose(self.basis_arrays_rt[ndim][0,...],
                                self.numpy_arrays_rt[ndim][0,...])
            self.assertAllclose(self.basis_arrays_rt[ndim][...,0],
                                self.numpy_arrays_rt[ndim][...,0])
            self.assertAllclose(self.basis_arrays_rt[ndim][0,...,0],
                                self.numpy_arrays_rt[ndim][0,...,0])
            self.assertAllclose(self.basis_arrays_rt[ndim][:,...],
                                self.numpy_arrays_rt[ndim][:,...])
            self.assertAllclose(self.basis_arrays_rt[ndim][...,:],
                                self.numpy_arrays_rt[ndim][...,:])
            self.assertAllclose(self.basis_arrays_rt[ndim][:,...,0],
                                self.numpy_arrays_rt[ndim][:,...,0])
            self.assertAllclose(self.basis_arrays_rt[ndim][0,...,:],
                                self.numpy_arrays_rt[ndim][0,...,:])
            self.assertAllclose(self.basis_arrays_rt[ndim][:,...,:],
                                self.numpy_arrays_rt[ndim][:,...,:])

    def test_newaxis(self):
        self.assertAllclose(self.a_nn[None].value, self.d_nn[None])
        self.assertTrue(self.a_nn[None].shape == self.d_nn[None].shape)
        self.assertAllclose(self.a_nn[:,None].value, self.d_nn[:,None])
        self.assertTrue(self.a_nn[:,None].shape == self.d_nn[:,None].shape)
        self.assertAllclose(self.a_nn[None,None].value, self.d_nn[None,None])
        self.assertTrue(self.a_nn[None,None].shape == self.d_nn[None,None].shape)


    def test_eigh(self):
        # NumPy
        e, v = np.linalg.eigh(self.dh_nn)
        self.assertAllclose(np.einsum('ai,i,bi->ab', v, e, v), self.dh_nn)
        self.assertAllclose(np.dot(v*e[None,:], v.T), self.dh_nn)
        # Basis Array
        e, v = basis.linalg.eigh(self.ah_nn)
        self.assertAllclose(basis.einsum('ai,i,bi->ab', v, e, v), self.ah_nn)
        #self.assertAllclose(np.dot(v*e[None,:], v.T), self.ah_nn)
        #v * e[None,:]


def loop_einsum_labels(ndim, start_label=0):
    indices = list(string.ascii_lowercase)[start_label:start_label+ndim]
    for nsum in range(0, ndim+1):
        for sumindices in itertools.combinations(range(ndim), nsum):
            contraction = indices.copy()
            for sumidx in sumindices:
                contraction[sumidx] = 'I'
            contraction = ''.join(contraction)
            yield contraction


def generate_test_1array(contraction, ndim):
    """Generator to avoid late-binding of contraction and ndim"""
    def test(self):
        self.assertAllclose(basis.einsum(contraction, self.basis_arrays_sq[ndim]),
                               np.einsum(contraction, self.numpy_arrays_sq[ndim]))
    return test


def generate_test_2array(contraction, ndim1, ndim2):
    """Generator to avoid late-binding of contraction and ndim"""
    def test(self):
        #print("RUNNING %s" % contraction)
        self.assertAllclose(basis.einsum(contraction, self.basis_arrays_sq[ndim1], self.basis_arrays_sq[ndim2]),
                               np.einsum(contraction, self.numpy_arrays_sq[ndim1], self.numpy_arrays_sq[ndim2]))
    return test


def generate_test_einsum_summation(cls, result=True):
    """Summation over one index in one array: abi->ab, aii->a, ..."""
    for ndim in range(1, 5):
        for labels in loop_einsum_labels(ndim):
            if result:
                rhs = '->' + labels.replace('I', '')
            else:
                rhs = ''
            contraction = labels + rhs
            func = generate_test_1array(contraction, ndim)
            funcname = 'test_einsum_summation_%s%s' % (labels, rhs.replace('->', '_to_'))
            setattr(cls, funcname, func)
            print("Adding function '%s'" % funcname)


def generate_test_einsum_contraction(cls, result=True):
    """Summation over one index in two arrays: ai,bi->ab, ..."""
    for ndim1, ndim2 in itertools.product(range(1, 5), repeat=2):
        for labels1 in loop_einsum_labels(ndim1):
            for labels2 in loop_einsum_labels(ndim2, start_label=ndim1):
                if result:
                    rhs = '->' + (labels1 + labels2).replace('I', '')
                else:
                    rhs = ''
                contraction = '%s,%s%s' % (labels1, labels2, rhs)
                func = generate_test_2array(contraction, ndim1, ndim2)
                funcname = 'test_einsum_contraction_%s_%s%s' % (labels1, labels2, rhs.replace('->', '_to_'))
                setattr(cls, funcname, func)
                print("Adding function '%s'" % funcname)


generate_test_einsum_summation(Tests)
# FIXME:
#generate_test_einsum_summation(Tests, False)
generate_test_einsum_contraction(Tests)


class DotTests(TestCase):

    def test_dot_11(self):
        n = 30
        a = np.random.rand(n)
        b = np.random.rand(n)
        c = np.dot(a, b)
        aa = basis.Array(a, basis=basis.nobasis)
        ab = basis.Array(b, basis=basis.nobasis)
        ac = basis.dot(aa, ab)
        self.assertAllclose(ac, c)

    def test_dot_21(self):
        n, m = 30, 40
        a = np.random.rand(n, m)
        b = np.random.rand(m)
        c = np.dot(a, b)
        aa = basis.Array(a, basis=(basis.nobasis, basis.nobasis))
        ab = basis.Array(b, basis=basis.nobasis)
        ac = basis.dot(aa, ab)
        self.assertAllclose(ac, c)

    def test_dot_31(self):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m, k)
        b = np.random.rand(k)
        c = np.dot(a, b)
        aa = basis.Array(a, basis=(basis.nobasis, basis.nobasis, basis.nobasis))
        ab = basis.Array(b, basis=basis.nobasis)
        ac = basis.dot(aa, ab)
        self.assertAllclose(ac, c)

    def test_dot_22(self):
        n, m, k = 30, 40, 50
        a = np.random.rand(n, m)
        b = np.random.rand(m, k)
        c = np.dot(a, b)
        aa = basis.Array(a, basis=(basis.nobasis, basis.nobasis))
        ab = basis.Array(b, basis=(basis.nobasis, basis.nobasis))
        ac = basis.dot(aa, ab)
        self.assertAllclose(ac, c)

    def test_dot_32(self):
        n, m, k, l = 30, 40, 50, 60
        a = np.random.rand(n, m, k)
        b = np.random.rand(k, l)
        c = np.dot(a, b)
        aa = basis.Array(a, basis=(basis.nobasis, basis.nobasis, basis.nobasis))
        ab = basis.Array(b, basis=(basis.nobasis, basis.nobasis))
        ac = basis.dot(aa, ab)
        self.assertAllclose(ac, c)


class EinsumTests(TestCase):

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
