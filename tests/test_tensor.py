import pytest
import numpy as np

import btensor
from testing import TensorTests, rand_orth_mat


class TestTensor(TensorTests):

    @classmethod
    def setup_class(cls) -> None:
        n = 10
        np.random.seed(0)
        cls.data = np.random.random((n, n))

    def test_not_writeable(self):
        data_backup = self.data.copy()

        # Perform copy
        tensor = self.tensor_cls(self.data)
        assert np.all(tensor == data_backup)
        self.data[:] = 0
        assert np.all(tensor == data_backup)
        self.data = data_backup

        # Do not copy
        tensor = self.tensor_cls(self.data, copy_data=False)
        assert np.all(tensor == data_backup)
        with pytest.raises(ValueError):
            self.data[:] = 0
        tensor._data.flags.writeable = True
        tensor._data[:] = 0
        assert np.all(tensor == 0)
        self.data = data_backup

    def test_copy(self):
        tensor = self.tensor_cls(self.data)
        tensor_copy = tensor.copy()
        tensor._data.flags.writeable = True
        tensor._data[:] = 0
        assert np.all(tensor_copy == self.data)


class TestArithmetricSameBasis(TensorTests):

    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        b1 = btensor.Basis(n1)
        b2 = btensor.Basis(n2)
        cls.d1 = np.random.rand(n1, n2)
        cls.d2 = np.random.rand(n1, n2)
        cls.a1 = cls.tensor_cls(cls.d1, basis=(b1, b2))
        cls.a2 = cls.tensor_cls(cls.d2, basis=(b1, b2))

    def test_negation(self):
        self.assert_allclose((-self.a1)._data, -self.d1)

    def test_absolute(self):
        self.assert_allclose(abs(self.a1)._data, abs(self.d1))

    def test_addition(self):
        self.assert_allclose((self.a1 + self.a2)._data, self.d1 + self.d2)

    def test_subtraction(self):
        self.assert_allclose((self.a1 - self.a2)._data, self.d1 - self.d2)

    def test_multiplication(self):
        self.assert_allclose((self.a1 * self.a2)._data, self.d1 * self.d2)

    def test_division(self):
        self.assert_allclose((self.a1 / self.a2)._data, self.d1 / self.d2)

    def test_floordivision(self):
        self.assert_allclose((self.a1 // self.a2)._data, self.d1 // self.d2)


class TestArithmetricDifferentBasis(TensorTests):

    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        n11, n12 = 3, 4
        n21, n22 = 6, 2
        b1 = btensor.Basis(n1)
        b2 = btensor.Basis(n2)
        cls.c11 = rand_orth_mat(n1, n11)
        cls.c12 = rand_orth_mat(n1, n12)
        cls.c21 = rand_orth_mat(n2, n21)
        cls.c22 = rand_orth_mat(n2, n22)
        b11 = btensor.Basis(cls.c11, parent=b1)
        b12 = btensor.Basis(cls.c12, parent=b1)
        b21 = btensor.Basis(cls.c21, parent=b2)
        b22 = btensor.Basis(cls.c22, parent=b2)
        cls.d1 = np.random.rand(n11, n21)
        cls.d2 = np.random.rand(n12, n22)
        cls.a1 = cls.tensor_cls(cls.d1, basis=(b11, b21))
        cls.a2 = cls.tensor_cls(cls.d2, basis=(b12, b22))
        # To check
        cls.t1 = np.linalg.multi_dot((cls.c11, cls.d1, cls.c21.T))
        cls.t2 = np.linalg.multi_dot((cls.c12, cls.d2, cls.c22.T))

    # --- Scalar

    def test_scalar_addition(self):
        self.assert_allclose((self.a1 + 2)._data, self.d1 + 2)
        self.assert_allclose((2 + self.a1)._data, self.d1 + 2)

    def test_scalar_subtraction(self):
        self.assert_allclose((self.a1 - 2)._data, self.d1 - 2)
        self.assert_allclose((2 - self.a1)._data, 2 - self.d1)

    def test_scalar_multiplication(self):
        self.assert_allclose((self.a1 * 2)._data, self.d1 * 2)
        self.assert_allclose((2 * self.a1)._data, 2 * self.d1)

    def test_scalar_division(self):
        self.assert_allclose((self.a1 / 2)._data, self.d1 / 2)
        self.assert_allclose((2 / self.a1)._data, 2 / self.d1)

    def test_scalar_floor_division(self):
        val = 0.02
        self.assert_allclose((self.a1 // val)._data, self.d1 // val)
        self.assert_allclose((val // self.a1)._data, val // self.d1)

    def test_scalar_power(self):
        val = 0.02
        self.assert_allclose((self.a1 // val)._data, self.d1 // val)
        self.assert_allclose((val // self.a1)._data, val // self.d1)

    # --- Other

    def test_addition(self):
        self.assert_allclose((self.a1 + self.a2)._data, self.t1 + self.t2)

    def test_subtraction(self):
        self.assert_allclose((self.a1 - self.a2)._data, self.t1 - self.t2)

    def test_multiplication(self):
        self.assert_allclose((self.a1 * self.a2)._data, self.t1 * self.t2)

    def test_division(self):
        self.assert_allclose((self.a1 / self.a2)._data, self.t1 / self.t2)

    def test_floor_division(self):
        self.assert_allclose((self.a1 // self.a2)._data, self.t1 // self.t2)

    def test_power(self):
        self.assert_allclose((self.a1 ** self.a2)._data, self.t1 ** self.t2)

    #@unittest.skip("TODO")
    #def test_iadd(self):
    #    a1 = self.a1
    #    a1 += 0
    #    self.assertEqual(id(a1), id(self.a1))


class TestSubbasis(TensorTests):

    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        m1, m2 = 3, 4
        b1 = btensor.Basis(n1)
        b2 = btensor.Basis(n2)
        cls.c1 = rand_orth_mat(n1, m1)
        cls.c2 = rand_orth_mat(n2, m2)
        sb1 = btensor.Basis(cls.c1, parent=b1)
        sb2 = btensor.Basis(cls.c2, parent=b2)
        cls.d1 = np.random.rand(n1, n2)
        cls.d2 = np.linalg.multi_dot((cls.c1.T, cls.d1, cls.c2))
        cls.a1 = cls.tensor_cls(cls.d1, basis=(b1, b2))
        cls.a2 = cls.tensor_cls(cls.d2, basis=(sb1, sb2))

    #def test_basis_setter(self):
    #    a1a = self.a1.project_onto(self.a2.basis)
    #    a1b = self.a1.copy()
    #    a1b.basis = self.a2.basis
    #    self.assertAllclose(a1a, a1b)

    def test_subspace(self):
        a2 = self.a1.proj(self.a2.basis)
        self.assert_allclose(self.a2._data, a2._data)
