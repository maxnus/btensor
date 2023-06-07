#import pytest
#import numpy as np
#from helper import TestCase1Array
#from btensor import Array
#from conftest import get_permutations_of_combinations
#
#
#@pytest.mark.skip("SKIP")
#class TestGetitem(TestCase1Array):
#
#    @pytest.mark.parametrize('element', [0, -1])
#    def test_getitem_int(self, array, ndim, element):
#        array, np_array = array
#        self.assert_allclose(array[element], np_array[element])
#        for dim in range(ndim):
#            key = dim*(slice(None), ) + (element,)
#            self.assert_allclose(array[key], np_array[key])
#
#    @pytest.fixture(params=[slice(None),
#                            slice(1), slice(2),
#                            slice(None, None, 1), slice(None, None, 2), slice(None, None, -1)],
#                    ids=lambda x: str(x))
#    def slice_(self, request):
#        return request.param
#
#    def test_getitem_slice(self, slice_):
#        self.assert_allclose(self.array[slice_], self.data[slice_])
#
#    def test_getitem_newaxis(self, array):
#        array, np_array = array
#        self.assert_allclose(array[np.newaxis], np_array[np.newaxis])
#
#    #@pytest.mark.parametrize('key', [[0], [0, 2], [-2, 1]])
#    #@pytest.mark.parametrize('keytype', [list, np.asarray])
#    #def test_getitem_list_array(self, ndim_atleast2, get_array_large, key, keytype):
#    #    array, np_array = get_array_large(ndim=ndim_atleast2)
#    #    #print(array.shape)
#    #    self.assert_allclose(array[keytype(key)], np_array[keytype(key)])
#
#    #def test_foo(self, array_large):
#    #    array_large = array_large[0]
#    #    print(array_large.shape)
#
#    #def test_getitem_list_array(self):
#    #    self.assert_allclose(self.array[[0]], self.data[[0]])
#    #    self.assert_allclose(self.array[np.asarray([0])], self.data[np.asarray([0])])
#    #    self.assert_allclose(self.array[[0, 2, 1]], self.data[[0, 2, 1]])
#    #    self.assert_allclose(self.array[np.asarray([0, 2, 1])], self.data[np.asarray([0, 2, 1])])
#    #    self.assert_allclose(self.array[[-2, 1]], self.data[[-2, 1]])
#    #    self.assert_allclose(self.array[np.asarray([-2, 1])], self.data[np.asarray([-2, 1])])
#
#    @pytest.mark.parametrize('key', [(0, ...), (..., 0), (0, 0, ...), (0, ..., 0), (..., 0, 0),
#                                     (slice(None), ...), (..., slice(None)), (slice(None), slice(None), ...),
#                                     (slice(None), ..., slice(None)), (..., slice(None), slice(None)),
#                                     (0, slice(None), ...), (0, ..., slice(None)), (..., 0, slice(None)),
#                                     (slice(None), 0, ...), (slice(None), ..., 0), (..., slice(None), 0)],
#                             ids=lambda x: str(x).replace('slice(None, None, None)', ':').replace('Ellipsis', '...'))
#    def test_getitem_tuple_with_ellipsis(self, ndim_atleast2, get_array, key):
#        array, np_array = get_array(ndim=ndim_atleast2)
#        self.assert_allclose(array[key], np_array[key])
#
#    @pytest.mark.parametrize('key', get_permutations_of_combinations([np.newaxis, slice(None)], 2),
#                             ids=lambda x: str(x))
#    def test_getitem_newaxis_tuple(self, ndim_atleast2, get_array, key):
#        array, np_array = get_array(ndim_atleast2)
#        self.assert_allclose(array[key], np_array[key])
#
#    @pytest.mark.parametrize('key', get_permutations_of_combinations([0, 1, -1, slice(None), slice(2), slice(1, None)],
#                                                                     minsize=1, maxsize=2), ids=lambda x: str(x))
#    def test_getitem_tuple(self, key, array_large_atleast2d):
#        array, np_array, basis = array_large_atleast2d
#        self.assert_allclose(array[key], np_array[key])
#
#    #@pytest.mark.parametrize('key', get_
#    #def test_getitem_tuple(self):
#    #
#    #
#    #    self.assert_allclose(self.array[0, 2], self.data[0, 2])
#    #    self.assert_allclose(self.array[-1, 2], self.data[-1, 2])
#    #    self.assert_allclose(self.array[:2, 3:]._data, self.data[:2, 3:])
#    #    self.assert_allclose(self.array[::2, ::-1]._data, self.data[::2, ::-1])
#    #    self.assert_allclose(self.array[0, :2]._data, self.data[0, :2])
#    #    self.assert_allclose(self.array[::-1, -1]._data, self.data[::-1, -1])
#    #    self.assert_allclose(self.array[..., 0], self.data[..., 0])
#    #    self.assert_allclose(self.array[..., :2], self.data[..., :2])
#    #    self.assert_allclose(self.array[::-1, ...], self.data[::-1, ...])
#    #    self.assert_allclose(self.array[np.newaxis, 0], self.data[np.newaxis, 0])
#    #    self.assert_allclose(self.array[:2, np.newaxis], self.data[:2, np.newaxis])
#
#    #def test_getitem_boolean(self):
#    #    mask = [True, True, False, False, True]
#    #    self.assertAllclose(self.a1[mask].value, self.d1[mask])
#
#    def test_setitem(self):
#        def test(key, value):
#            a1 = self.a1.copy()
#            d1 = self.d1.copy()
#            a1[key] = value
#            d1[key] = value
#            self.assert_allclose(a1, d1)
#
#        test(0, 0)
#        test(slice(None), 0)
#        test(slice(1, 2), 0)
#        test(..., 0)
#        test([1, 2], 0)
#        test(([1, 2, -1], [0, 3, 2]), 0)
#        test((0, 2), 0)
#        test((slice(None), 2), 0)
#        test((slice(1, 2), slice(3, 0, -1)), 0)
#        test((slice(1, 2), [0, 2]), 0)
#        test(np.newaxis, 0)