#     Copyright 2023 Max Nusspickel
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import operator

import pytest
import numpy as np

from helper import TestCase
from conftest import subbasis_definition_to_matrix

from btensor import Tensor
from btensor.util import BasisDependentOperationError


class TestTensor(TestCase):

    def test_data_copy(self, tensor_cls, np_array):
        data = np_array.copy()
        tensor = tensor_cls(data)
        data[:] = 0
        assert np.all(tensor.to_numpy() == np_array)

    def test_tensor_copy(self, tensor_cls, np_array):
        tensor = tensor_cls(np_array)
        tensor_copy = tensor.copy()
        tensor_copy._data.flags.writeable = True
        tensor_copy._data[:] = 0
        assert np.all(tensor.to_numpy() == np_array)

    def test_data_copy_nocopy(self, tensor_cls, np_array):
        data = np_array.copy()
        tensor = tensor_cls(data, copy_data=False)
        data.flags.writeable = True
        data[:] = 0
        assert np.all(tensor.to_numpy() == 0)

    def test_tensor_for_loop_raises(self, np_array):
        tensor = Tensor(np_array)
        with pytest.raises(TypeError):
            for element in tensor:
                pass

    @pytest.mark.parametrize('inplace', [True, False])
    def test_replace_basis_inplace(self, tensor_or_array, inplace):
        tensor = tensor_or_array[0]
        tensor_out = tensor.replace_basis((None,), inplace=inplace)
        if inplace:
            assert id(tensor_out) == id(tensor)
        else:
            assert id(tensor_out) != id(tensor)
        assert tensor_out.basis == tensor.basis


class TestArithmetic(TestCase):

    @pytest.mark.parametrize('unary_operator', [operator.pos, operator.neg])
    def test_unary_operator(self, unary_operator, tensor_or_array):
        tensor, np_array = tensor_or_array
        self.assert_allclose(unary_operator(tensor), unary_operator(np_array))

    def test_unary_operator_exception(self, tensor_or_array):
        tensor, np_array = tensor_or_array
        with pytest.raises(BasisDependentOperationError):
            self.assert_allclose(abs(tensor), abs(np_array))

    @pytest.mark.parametrize('binary_operator', [operator.add, operator.sub])
    def test_binary_operator(self, ndim, tensor_cls, binary_operator, get_tensor_or_array):
        (tensor1, np_array1), (tensor2, np_array2) = get_tensor_or_array(ndim, tensor_cls, number=2)
        self.assert_allclose(binary_operator(tensor1, tensor2), binary_operator(np_array1, np_array2))

    @pytest.mark.parametrize('scalar', [-2.2, -1, -0.4, 0, 0.3, 1, 1.2, 2, 3.3])
    @pytest.mark.parametrize('binary_operator', [operator.add, operator.sub, operator.mul, operator.truediv])
    def test_scalar_operator(self, scalar, ndim, tensor_cls, binary_operator, tensor_or_array):
        tensor, np_array = tensor_or_array
        expected = binary_operator(np_array, scalar)
        result = binary_operator(tensor, scalar)
        self.assert_allclose(result, expected)

    @pytest.mark.parametrize('scalar', [-2.2, -1, -0.4, 0, 0.3, 1, 1.2, 2, 3.3])
    @pytest.mark.parametrize('binary_operator', [operator.add, operator.sub])
    def test_scalar_operator_reverse(self, scalar, ndim, tensor_cls, binary_operator, tensor_or_array):
        tensor, np_array = tensor_or_array
        expected = binary_operator(scalar, np_array)
        result = binary_operator(scalar, tensor)
        self.assert_allclose(result, expected)

    @pytest.mark.parametrize('subsize1', [1, 5, 10])
    @pytest.mark.parametrize('subsize2', [1, 5, 10])
    @pytest.mark.parametrize('binary_operator', [operator.add, operator.sub])
    def test_binary_operator_different_basis(self, binary_operator, subsize1, subsize2, subbasis_type_2x, tensor_cls_2x,
                                             get_rootbasis_subbasis):
        subtype1, subtype2 = subbasis_type_2x
        rootbasis, (subbasis1, subarg1), (subbasis2, subarg2) = get_rootbasis_subbasis(10, subsize1, subtype1, subsize2,
                                                                                       subtype2)
        np.random.seed(0)
        np_array1 = np.random.random((rootbasis.size, subbasis1.size, subbasis1.size))
        np_array2 = np.random.random((rootbasis.size, subbasis2.size, subbasis2.size))
        tensor_cls1, tensor_cls2 = tensor_cls_2x
        tensor1 = tensor_cls1(np_array1, basis=(rootbasis, subbasis1, subbasis1))
        tensor2 = tensor_cls2(np_array2, basis=(rootbasis, subbasis2, subbasis2))

        subarg1 = subbasis_definition_to_matrix(subarg1, rootbasis.size)
        subarg2 = subbasis_definition_to_matrix(subarg2, rootbasis.size)

        expected = binary_operator(np.einsum('xab,ia,jb->xij', np_array1, subarg1, subarg1),
                                   np.einsum('xab,ia,jb->xij', np_array2, subarg2, subarg2))
        if binary_operator in {operator.truediv, operator.pow}:
            rtol = 1e-10
        else:
            rtol = self.allclose_rtol
        self.assert_allclose(binary_operator(tensor1, tensor2), expected, rtol=rtol)


# class TestSubbasis(TestCase):
#
#     @classmethod
#     def setup_class(cls):
#        np.random.seed(0)
#        n1, n2 = 5, 6
#        m1, m2 = 3, 4
#        b1 = btensor.Basis(n1)
#        b2 = btensor.Basis(n2)
#        cls.c1 = rand_orth_mat(n1, m1)
#        cls.c2 = rand_orth_mat(n2, m2)
#        sb1 = btensor.Basis(cls.c1, parent=b1)
#        sb2 = btensor.Basis(cls.c2, parent=b2)
#        cls.d1 = np.random.rand(n1, n2)
#        cls.d2 = np.linalg.multi_dot((cls.c1.T, cls.d1, cls.c2))
#        cls.a1 = cls.tensor_cls(cls.d1, basis=(b1, b2))
#        cls.a2 = cls.tensor_cls(cls.d2, basis=(sb1, sb2))
#
#     def test_basis_setter(self):
#        a1a = self.a1.project_onto(self.a2.basis)
#        a1b = self.a1.copy()
#        a1b.basis = self.a2.basis
#        self.assertAllclose(a1a, a1b)
#
#     def test_subspace(self):
#        a2 = self.a1.proj(self.a2.basis)
#        self.assert_allclose(self.a2._data, a2._data)


class TestGetitem(TestCase):

    @pytest.mark.parametrize('subsize', [6, 3, 1])
    def test_getitem(self, subsize, subbasis_type, get_rootbasis_subbasis, tensor_cls, ndim):
        rootsize = 6
        rootbasis, (subbasis, subarg) = get_rootbasis_subbasis(rootsize, subsize, subbasis_type)
        rootbasis = ndim*(rootbasis,)
        subbasis = ndim*(subbasis,)
        np_array = np.random.random(ndim*(rootsize,))
        tensor = tensor_cls(np_array, basis=rootbasis)
        self.assert_allclose(tensor[subbasis], tensor.project(subbasis))

    def test_getitem_slice_none(self, tensor_or_array):
        tensor, np_array = tensor_or_array
        self.assert_allclose(tensor[:], np_array[:])

    def test_getitem_ellipsis(self, tensor_or_array):
        tensor, np_array = tensor_or_array
        self.assert_allclose(tensor[...], np_array[...])

    @pytest.mark.parametrize('key', [0, slice(0, 100), [0], [True]], ids=lambda x: str(x))
    def test_getitem_raises(self, key, tensor):
        tensor = tensor[0]
        with pytest.raises(TypeError):
            assert tensor[key]

