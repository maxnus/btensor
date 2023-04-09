import unittest
import numpy as np

import basis_array as basis
from testing import TestCase, rand_orth_mat
from basis_array import util


class TestBasis(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.size_a, cls.size_b = 6, 6

        def metric(n):
            noise = 0.1*(np.random.random((n, n))-0.5)
            return np.identity(n) + noise + noise.T

        cls.metric_a = metric(cls.size_a)
        cls.metric_b = metric(cls.size_b)

        cls.rootbasis_a = ba = basis.B(cls.size_a, metric=cls.metric_a)
        cls.rootbasis_b = bb = basis.B(cls.size_b, metric=cls.metric_b)

        # Subbasis

        def make_subbasis(b, permutation=False):
            subbasis = []
            trafos = []
            parent = b
            while parent.size > 1:
                if permutation:
                    t = np.random.permutation(range(parent.size))[:parent.size-1]
                    #print(np.identity(parent.size)[:,t].shape)
                    trafos.append(np.identity(parent.size)[:,t])
                else:
                    t = rand_orth_mat(parent.size, parent.size-1)
                    trafos.append(t)
                b = basis.B(t, parent=parent)
                subbasis.append(b)
                parent = b
            return subbasis, trafos

        cls.subbasis_a, cls.trafos_a = make_subbasis(cls.rootbasis_a)
        cls.subbasis_b, cls.trafos_b = make_subbasis(cls.rootbasis_b, permutation=True)

        cls.basis_a = [cls.rootbasis_a, *cls.subbasis_a]
        cls.basis_b = [cls.rootbasis_b, *cls.subbasis_b]

    def test_is_root(self):
        self.assertTrue(self.rootbasis_a.is_root())
        self.assertTrue(self.rootbasis_b.is_root())
        for b in self.subbasis_a:
            self.assertFalse(b.is_root())
        for b in self.subbasis_b:
            self.assertFalse(b.is_root())

    def test_size(self):
        self.assertEqual(self.rootbasis_a.size, self.size_a)
        self.assertEqual(self.rootbasis_b.size, self.size_b)
        for i, b in enumerate(self.subbasis_a):
            self.assertEqual(b.size, self.rootbasis_a.size - (i + 1))

    def test_matrices_for_coeff_in_basis(self):

        def test(basis1, basis2, expected):
            mats = basis1.matrices_for_coeff_in_basis(basis2)
            self.assertAllclose(util.to_array(util.MatrixProduct(mats).evaluate()), expected)

        test(self.rootbasis_a, self.rootbasis_a, np.identity(self.rootbasis_a.size))
        test(self.rootbasis_b, self.rootbasis_b, np.identity(self.rootbasis_b.size))

        for b in self.subbasis_a:
            test(b, b, np.identity(b.size))
        for b in self.subbasis_b:
            test(b, b, np.identity(b.size))

        # Test all subbasis
        for i, b1 in enumerate(self.basis_a): # Subbasis
            for j, b2 in enumerate(self.basis_a[:i]): # Superbasis
                if i == j:
                    expected = np.identity(b1.size)
                else:
                    mats = self.trafos_a[j:i]
                    if len(mats) == 1:
                        expected = mats[0]
                    else:
                        expected = np.linalg.multi_dot(mats)
                test(b1, b2, expected)

        for i, b1 in enumerate(self.basis_b): # Subbasis
            for j, b2 in enumerate(self.basis_b[:i]): # Superbasis
                if i == j:
                    expected = np.identity(b1.size)
                else:
                    mats = self.trafos_b[j:i]
                    print([x.shape for x in mats])
                    if len(mats) == 1:
                        expected = mats[0]
                    else:
                        expected = np.linalg.multi_dot(mats)
                test(b1, b2, expected)



