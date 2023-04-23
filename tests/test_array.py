import unittest
import numpy as np

import btensor as basis
from testing import TestCase, rand_orth_mat


class ArithmetricTestsSameBasis(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        b1 = basis.Basis(n1)
        b2 = basis.Basis(n2)
        cls.d1 = np.random.rand(n1, n2)
        cls.d2 = np.random.rand(n1, n2)
        cls.a1 = basis.Tensor(cls.d1, basis=(b1, b2))
        cls.a2 = basis.Tensor(cls.d2, basis=(b1, b2))

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
        self.assertAllclose(self.a1[0, 2], self.d1[0, 2])
        self.assertAllclose(self.a1[-1, 2], self.d1[-1, 2])
        self.assertAllclose(self.a1[:2, 3:].value, self.d1[:2, 3:])
        self.assertAllclose(self.a1[::2, ::-1].value, self.d1[::2, ::-1])
        self.assertAllclose(self.a1[0, :2].value, self.d1[0, :2])
        self.assertAllclose(self.a1[::-1, -1].value, self.d1[::-1, -1])
        self.assertAllclose(self.a1[..., 0], self.d1[..., 0])
        self.assertAllclose(self.a1[..., :2], self.d1[..., :2])
        self.assertAllclose(self.a1[::-1, ...], self.d1[::-1, ...])


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
        b1 = basis.Basis(n1)
        b2 = basis.Basis(n2)
        cls.c11 = rand_orth_mat(n1, n11)
        cls.c12 = rand_orth_mat(n1, n12)
        cls.c21 = rand_orth_mat(n2, n21)
        cls.c22 = rand_orth_mat(n2, n22)
        b11 = basis.Basis(cls.c11, parent=b1)
        b12 = basis.Basis(cls.c12, parent=b1)
        b21 = basis.Basis(cls.c21, parent=b2)
        b22 = basis.Basis(cls.c22, parent=b2)
        cls.d1 = np.random.rand(n11, n21)
        cls.d2 = np.random.rand(n12, n22)
        cls.a1 = basis.Tensor(cls.d1, basis=(b11, b21))
        cls.a2 = basis.Tensor(cls.d2, basis=(b12, b22))
        # To check
        cls.t1 = np.linalg.multi_dot((cls.c11, cls.d1, cls.c21.T))
        cls.t2 = np.linalg.multi_dot((cls.c12, cls.d2, cls.c22.T))

    # --- Scalar

    def test_scalar_addition(self):
        self.assertAllclose((self.a1 + 2).value, self.d1 + 2)
        self.assertAllclose((2 + self.a1).value, self.d1 + 2)

    def test_scalar_subtraction(self):
        self.assertAllclose((self.a1 - 2).value, self.d1 - 2)
        self.assertAllclose((2 - self.a1).value, 2 - self.d1)\

    def test_scalar_multiplication(self):
        self.assertAllclose((self.a1 * 2).value, self.d1 * 2)
        self.assertAllclose((2 * self.a1).value, 2 * self.d1)

    def test_scalar_division(self):
        self.assertAllclose((self.a1 / 2).value, self.d1 / 2)
        self.assertAllclose((2 / self.a1).value, 2 / self.d1)

    def test_scalar_floor_division(self):
        val = 0.02
        self.assertAllclose((self.a1 // val).value, self.d1 // val)
        self.assertAllclose((val // self.a1).value, val // self.d1)

    def test_scalar_power(self):
        val = 0.02
        self.assertAllclose((self.a1 // val).value, self.d1 // val)
        self.assertAllclose((val // self.a1).value, val // self.d1)

    # --- Other

    def test_addition(self):
        self.assertAllclose((self.a1 + self.a2).value, self.t1 + self.t2)

    def test_subtraction(self):
        self.assertAllclose((self.a1 - self.a2).value, self.t1 - self.t2)

    def test_multiplication(self):
        self.assertAllclose((self.a1 * self.a2).value, self.t1 * self.t2)

    def test_division(self):
        self.assertAllclose((self.a1 / self.a2).value, self.t1 / self.t2)

    def test_floor_division(self):
        self.assertAllclose((self.a1 // self.a2).value, self.t1 // self.t2)

    def test_power(self):
        self.assertAllclose((self.a1 ** self.a2).value, self.t1 ** self.t2)

    @unittest.skip("TODO")
    def test_iadd(self):
        a1 = self.a1
        a1 += 0
        self.assertEqual(id(a1), id(self.a1))


class SubbasisTests(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        m1, m2 = 3, 4
        b1 = basis.Basis(n1)
        b2 = basis.Basis(n2)
        cls.c1 = rand_orth_mat(n1, m1)
        cls.c2 = rand_orth_mat(n2, m2)
        sb1 = basis.Basis(cls.c1, parent=b1)
        sb2 = basis.Basis(cls.c2, parent=b2)
        cls.d1 = np.random.rand(n1, n2)
        cls.d2 = np.linalg.multi_dot((cls.c1.T, cls.d1, cls.c2))
        cls.a1 = basis.Tensor(cls.d1, basis=(b1, b2))
        cls.a2 = basis.Tensor(cls.d2, basis=(sb1, sb2))

    #def test_basis_setter(self):
    #    a1a = self.a1.project_onto(self.a2.basis)
    #    a1b = self.a1.copy()
    #    a1b.basis = self.a2.basis
    #    self.assertAllclose(a1a, a1b)

    def test_subspace(self):
        a2 = self.a1.proj(self.a2.basis)
        self.assertAllclose(self.a2, a2)


if __name__ == '__main__':
    unittest.main()
