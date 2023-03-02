import unittest
import numpy as np
import scipy
import scipy.stats

import basis_array as basis


def rand_orth_mat(n, ncol=None):
    m = scipy.stats.ortho_group.rvs(n)
    #return m
    if ncol is not None:
        m = m[:,:ncol]
    return m


class TestCase(unittest.TestCase):

    allclose_atol = 1e-8
    allclose_rtol = 1e-7

    def assertAllclose(self, actual, desired, rtol=allclose_atol, atol=allclose_rtol, **kwargs):
        # Compare multiple pairs of arrays:
        if isinstance(actual, (tuple, list)):
            for i in range(len(actual)):
                self.assertAllclose(actual[i], desired[i], rtol=rtol, atol=atol, **kwargs)
            return
        # Compare single pair of arrays:
        try:
            np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, **kwargs)
        except AssertionError as e:
            # Add higher precision output:
            message = e.args[0]
            args = e.args[1:]
            message += '\nHigh precision:\n x: %r\n y: %r' % (actual, desired)
            e.args = (message, *args)
            raise

    def setUp(self):
        np.random.seed(0)


class ArithmetricTestsSameBasis(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        b1 = basis.B(n1)
        b2 = basis.B(n2)
        cls.d1 = np.random.rand(n1, n2)
        cls.d2 = np.random.rand(n1, n2)
        cls.a1 = basis.Array(cls.d1, basis=(b1, b2))
        cls.a2 = basis.Array(cls.d2, basis=(b1, b2))

    def test_negation(self):
        self.assertAllclose((-self.a1).value, -self.d1)

    def test_absolute(self):
        self.assertAllclose(abs(self.a1).value, abs(self.d1))

    def test_addition(self):
        self.assertAllclose((self.a1 + self.a2).value, self.d1+self.d2)

    def test_subtraction(self):
        self.assertAllclose((self.a1 - self.a2).value, self.d1-self.d2)

    def test_multiplication(self):
        self.assertAllclose((self.a1 * self.a2).value, self.d1 * self.d2)

    def test_division(self):
        self.assertAllclose((self.a1 / self.a2).value, self.d1 / self.d2)

    def test_floordivision(self):
        self.assertAllclose((self.a1 // self.a2).value, self.d1 // self.d2)

    def test_getitem_int(self):
        self.assertAllclose(self.a1[0].value, self.d1[0])
        self.assertAllclose(self.a1[-1].value, self.d1[-1])

    def test_getitem_slice(self):
        self.assertAllclose(self.a1[:2].value, self.d1[:2])
        self.assertAllclose(self.a1[::2].value, self.d1[::2])
        self.assertAllclose(self.a1[:-1].value, self.d1[:-1])

    def test_getitem_elipsis(self):
        self.assertAllclose(self.a1[...].value, self.d1[...])

    def test_getitem_tuple(self):
        self.assertAllclose(self.a1[0,2], self.d1[0,2])
        self.assertAllclose(self.a1[-1,2], self.d1[-1,2])
        self.assertAllclose(self.a1[:2,3:].value, self.d1[:2,3:])
        self.assertAllclose(self.a1[::2,::-1].value, self.d1[::2,::-1])
        self.assertAllclose(self.a1[0,:2].value, self.d1[0,:2])
        self.assertAllclose(self.a1[::-1,-1].value, self.d1[::-1,-1])
        self.assertAllclose(self.a1[...,0], self.d1[...,0])
        self.assertAllclose(self.a1[...,:2], self.d1[...,:2])
        self.assertAllclose(self.a1[::-1,...], self.d1[::-1,...])

    #def test_getitem_boolean(self):
    #    mask = [True, True, False, False, True]
    #    self.assertAllclose(self.a1[mask].value, self.d1[mask])


class ArithmetricTestsDifferentBasis(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        n11, n12 = 3, 4
        n21, n22 = 6, 2
        b1 = basis.B(n1)
        b2 = basis.B(n2)
        cls.c11 = rand_orth_mat(n1, n11)
        cls.c12 = rand_orth_mat(n1, n12)
        cls.c21 = rand_orth_mat(n2, n21)
        cls.c22 = rand_orth_mat(n2, n22)
        b11 = basis.B(b1, cls.c11)
        b12 = basis.B(b1, cls.c12)
        b21 = basis.B(b2, cls.c21)
        b22 = basis.B(b2, cls.c22)
        cls.d1 = np.random.rand(n11, n21)
        cls.d2 = np.random.rand(n12, n22)
        cls.a1 = basis.Array(cls.d1, basis=(b11, b21))
        cls.a2 = basis.Array(cls.d2, basis=(b12, b22))
        # To check
        cls.t1 = np.linalg.multi_dot((cls.c11, cls.d1, cls.c21.T))
        cls.t2 = np.linalg.multi_dot((cls.c12, cls.d2, cls.c22.T))

    def test_addition(self):
        self.assertAllclose((self.a1 + self.a2).value, self.t1 + self.t2)

    def test_subtraction(self):
        self.assertAllclose((self.a1 - self.a2).value, self.t1 - self.t2)

    def test_multiplication(self):
        self.assertAllclose((self.a1 * self.a2).value, self.t1 * self.t2)

    def test_division(self):
        self.assertAllclose((self.a1 / self.a2).value, self.t1 / self.t2)


class SubbasisTests(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        m1, m2 = 3, 4
        b1 = basis.B(n1)
        b2 = basis.B(n2)
        cls.c1 = rand_orth_mat(n1, m1)
        cls.c2 = rand_orth_mat(n2, m2)
        sb1 = basis.B(b1, cls.c1)
        sb2 = basis.B(b2, cls.c2)
        cls.d1 = np.random.rand(n1, n2)
        cls.d2 = np.linalg.multi_dot((cls.c1.T, cls.d1, cls.c2))
        cls.a1 = basis.Array(cls.d1, basis=(b1, b2))
        cls.a2 = basis.Array(cls.d2, basis=(sb1, sb2))

    def test_basis_setter(self):
        a1a = self.a1.as_basis(self.a2.basis)
        a1b = self.a1.copy()
        a1b.basis = self.a2.basis
        self.assertAllclose(a1a, a1b)

    def test_subspace(self):
        a2 = self.a1.as_basis(self.a2.basis)
        self.assertAllclose(self.a2, a2)


if __name__ == '__main__':
    unittest.main()
