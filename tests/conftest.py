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

from __future__ import annotations
import operator

import pytest
import itertools
import numpy as np
import scipy

import btensor
from btensor import Basis, Array, Tensor, TensorSum


class UserSlice:

    def __init__(self, start=None, stop=None, step=None):
        self.start = start
        self.stop = stop
        self.step = step

    def __repr__(self):
        return f"{type(self).__name__}({self.start}, {self.stop}, {self.step})"

    @classmethod
    def from_slice(cls, slc: slice) -> UserSlice:
        return cls(slc.start, slc.stop, slc.step)

    def to_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)


def variable_sized_product(values, minsize=1, maxsize=None):
    if maxsize is None:
        maxsize = len(values)
    values = [UserSlice.from_slice(v) if isinstance(v, slice) else v for v in values]
    output = []
    for size in range(minsize, maxsize+1):
        #combinations = list(itertools.combinations_with_replacement(values, size))
        #for comb in combinations:
        #    perms = set(itertools.permutations(comb))
        #    output += list(perms)
        output += list(itertools.product(values, repeat=size))
    output = [tuple(y.to_slice() if isinstance(y, UserSlice) else y for y in x) for x in output]
    return output


def random_orthogonal_matrix(n, ncolumn=None, rng=np.random.default_rng()):
    if n == 1:
        return np.asarray([[1.0]])[:, :ncolumn]
    m = scipy.stats.ortho_group.rvs(n, random_state=rng)
    if ncolumn is not None:
        m = m[:, :ncolumn]
    return m


def powerset(iterable, include_empty=True):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    start = 0 if include_empty else 1
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(start, len(s)+1))


def get_ndims_and_axes(mindim=1, maxdim=4):
    ndims_and_axes = []
    for dim in range(mindim, maxdim+1):
        axes = list(powerset(range(dim), include_empty=False))
        ndims_and_axes += list(zip(len(axes)*[dim], axes))
    return ndims_and_axes


def get_ndims_and_same_size_axes(mindim=1, maxdim=4):
    ndims_and_axes = []
    for dim in range(mindim, maxdim+1):
        axes = list(itertools.permutations(range(dim)))
        ndims_and_axes += list(zip(len(axes)*[dim], axes))
    return ndims_and_axes


def get_ndims_and_axis12(mindim=1, maxdim=4):
    ndims_and_axis12 = []
    for dim in range(mindim, maxdim+1):
        axes = itertools.permutations(range(dim), 2)
        axes = list(zip(*axes))
        axes1 = axes[0] if axes else []
        axes2 = axes[1] if axes else []
        ndims_and_axis12 += list(zip(len(axes1)*[dim], axes1, axes2))
    return ndims_and_axis12


# --- Root fixtures

@pytest.fixture(params=range(3), scope='module', ids=lambda x: f'seed{x}')
def rng(request):
    return np.random.default_rng(request.param)


@pytest.fixture(params=[1, 2, 3, 4], scope='module', ids=lambda x: f'ndim{x}')
def ndim(request):
    return request.param


@pytest.fixture(params=[2, 3, 4], scope='module', ids=lambda x: f'ndim{x}')
def ndim_atleast2(request):
    return request.param


@pytest.fixture(params=get_ndims_and_axes(), scope='module', ids=lambda x: f'ndim{x[0]}-axes{x[1]}')
def ndim_and_axis(request):
    return request.param


@pytest.fixture(params=get_ndims_and_same_size_axes(), scope='module', ids=lambda x: f'ndim{x[0]}-axes{x[1]}')
def ndim_and_same_size_axis(request):
    return request.param


@pytest.fixture(params=get_ndims_and_axis12(), scope='module', ids=lambda x: f'ndim{x[0]}-axes{x[1]}')
def ndim_axis1_axis2(request):
    return request.param


@pytest.fixture(params=[Tensor, Array], scope='module')
def tensor_cls(request):
    """One tensor class (Tensor or Array)"""
    return request.param


@pytest.fixture(params=[1, 4], scope='module', ids=lambda x: f'size{x}')
def basis_size(request):
    return request.param


#@pytest.fixture(params=[4], scope='module', ids=lambda x: f'size{x}')
#def basis_size_large(request):
#    return request.param


@pytest.fixture(params=[operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.pow],
                scope='module')
def binary_operator(request):
    return request.param

# --- Combi fixtures


@pytest.fixture(params=[Tensor, Array], scope='module')
def tensor_cls_2x(request, tensor_cls):
    """All combination of two tensor classes (Tensor, Tensor), (Tensor, Array), ..."""
    return tensor_cls, request.param


@pytest.fixture(params=[Tensor, Array], scope='module')
def tensor_cls_3x(request, tensor_cls_2x):
    """All combination of two tensor classes (Tensor, Tensor, Tensor), (Tensor, Tensor, Array), ..."""
    return *tensor_cls_2x, request.param


# --- Derived fixtures


@pytest.fixture(scope='module')
def rootbasis(basis_size):
    return Basis(basis_size)


#@pytest.fixture(scope='module')
#def rootbasis_large(basis_size_large):
#    return Basis(basis_size_large)


@pytest.fixture(params=[0, -1], scope='module')
def subbasis(request, rootbasis):
    size = max(rootbasis.size + request.param, 1)
    return rootbasis.make_subbasis(random_orthogonal_matrix(rootbasis.size, ncolumn=size))


@pytest.fixture(params=[(0, 0), (-1, 0), (0, -1), (-1, -1)], scope='module')
def subbasis_2x(request, rootbasis):
    n = rootbasis.size
    size1 = n + request.param[0]
    size2 = n + request.param[1]
    basis1 = rootbasis.make_subbasis(random_orthogonal_matrix(n, ncolumn=size1))
    basis2 = rootbasis.make_subbasis(random_orthogonal_matrix(n, ncolumn=size2))
    return basis1, basis2


@pytest.fixture(params=['rotation', 'indices', 'slice', 'mask'], scope='module')
def subbasis_type(request):
    return request.param


@pytest.fixture(params=['rotation', 'indices', 'slice', 'mask'], scope='module')
def subbasis_type_2x(request, subbasis_type):
    return request.param, subbasis_type


def get_random_subbasis_definition(rootsize, subsize, subtype, rng=np.random.default_rng()):
    if subtype == 'rotation':
        return random_orthogonal_matrix(rootsize, ncolumn=subsize, rng=rng)
    if subtype in ('indices', 'mask'):
        r = rng.permutation(range(rootsize))[:subsize]
        if subtype == 'mask':
            r = np.isin(np.arange(rootsize), r)
        return r
    if subtype == 'slice':
        size = rootsize - subsize + 1
        assert size > 0
        start = rng.integers(0, size)
        stop = start + subsize
        return slice(start, stop, 1)
    raise ValueError(subtype)


def subbasis_definition_to_matrix(subarg, rootsize):
    if getattr(subarg, 'ndim', None) == 2:
        return subarg
    return np.identity(rootsize)[:, subarg]


@pytest.fixture(scope='module')
def get_rootbasis_subbasis():
    def get_rootbasis_subbasis(rootsize, subsize, subtype, subsize2=None, subtype2=None):
        if subsize > rootsize:
            raise ValueError
        rootbasis = Basis(rootsize)
        subarg = get_random_subbasis_definition(rootsize, subsize, subtype)
        subbasis = rootbasis.make_subbasis(subarg)
        if subsize2 is not None and subtype2 is not None:
            subarg2 = get_random_subbasis_definition(rootsize, subsize2, subtype2)
            subbasis2 = rootbasis.make_subbasis(subarg2)
            return rootbasis, (subbasis, subarg), (subbasis2, subarg2)
        return rootbasis, (subbasis, subarg)
    return get_rootbasis_subbasis


@pytest.fixture(params=variable_sized_product([0, 1, 3]), scope='module',
                ids=lambda x: f'shape' + ''.join([str(y) for y in x]))
def shape_incl_empty(request):
    return request.param


@pytest.fixture(params=variable_sized_product([1, 3], maxsize=4), scope='module',
                ids=lambda x: f'shape' + ''.join([str(y) for y in x]))
def shape(request):
    return request.param


@pytest.fixture(params=variable_sized_product([4, 5], maxsize=4), scope='module',
                ids=lambda x: f'shape' + ''.join([str(y) for y in x]))
def shape_large(request):
    return request.param


@pytest.fixture(params=variable_sized_product([4, 5], minsize=2, maxsize=4), scope='module',
                ids=lambda x: f'shape' + ''.join([str(y) for y in x]))
def shape_large_atleast2d(request):
    return request.param


@pytest.fixture(scope='module')
def np_array(shape):
    return np.random.random(shape)


@pytest.fixture(scope='module')
def np_array_large(shape_large):
    return np.random.random(shape_large)


@pytest.fixture(scope='module')
def np_array_large_atleast2d(shape_large_atleast2d):
    return np.random.random(shape_large_atleast2d)


@pytest.fixture
def basis_for_shape(shape):
    return tuple(Basis(size) for size in shape)


@pytest.fixture(params=variable_sized_product([1, 3, -1, -3], maxsize=4), scope='module',
                ids=lambda x: f'shape' + ''.join([str(y) for y in x]))
def shape_and_basis(request):
    shape = tuple(abs(size) for size in request.param)
    basis = tuple(Basis(size) if size > 0 else btensor.nobasis for size in request.param)
    return shape, basis


@pytest.fixture
def basis_for_shape_large_atleast2d(shape_large_atleast2d):
    return tuple(Basis(size) for size in shape_large_atleast2d)


@pytest.fixture
def array_large_atleast2d(shape_large_atleast2d, basis_for_shape_large_atleast2d):
    basis = basis_for_shape_large_atleast2d
    np_array = np.random.random(shape_large_atleast2d)
    array = Array(np_array, basis=basis)
    return array, np_array, basis


@pytest.fixture(scope='module')
def get_tensor_or_array(rootbasis):
    def get_tensor_or_array(ndim: int, tensor_cls: type, number: int = 1, hermitian: bool = False) \
            -> list[tuple] | tuple:
        np.random.seed(0)
        basis = tuple(ndim * [rootbasis])
        result = []
        for n in range(number):
            data = np.random.random(tuple([b.size for b in basis]))
            if hermitian:
                data = (data + data.T)/2
            tensor = tensor_cls(data, basis=basis)
            result.append((tensor, data))
        if number == 1:
            return result[0]
        return result
    return get_tensor_or_array


@pytest.fixture(scope='module')
def get_tensor(get_tensor_or_array):
    def get_tensor(ndim, number=1, hermitian=False):
        return get_tensor_or_array(ndim, Tensor, number=number, hermitian=hermitian)
    return get_tensor


@pytest.fixture(scope='module')
def get_array(get_tensor_or_array):
    def get_array(ndim, number=1, hermitian=False):
        return get_tensor_or_array(ndim, Array, number=number, hermitian=hermitian)
    return get_array


@pytest.fixture(scope='module')
def tensor_or_array(ndim, tensor_cls, get_tensor_or_array):
    return get_tensor_or_array(ndim, tensor_cls)


@pytest.fixture(scope='module')
def tensor(ndim, get_tensor):
    return get_tensor(ndim)


@pytest.fixture(scope='module')
def array(ndim, get_array):
    return get_array(ndim)


@pytest.fixture(scope='module')
def tensor_2x(tensor_cls_2x, ndim, rootbasis):
    np.random.seed(0)
    tensor_cls1, tensor_cls2 = tensor_cls_2x
    basis = tuple(ndim * [rootbasis])
    data1 = np.random.random(tuple([b.size for b in basis]))
    data2 = np.random.random(tuple([b.size for b in basis]))
    tensor1 = tensor_cls1(data1, basis=basis)
    tensor2 = tensor_cls1(data2, basis=basis)
    return (tensor1, data1), (tensor2, data2)


@pytest.fixture(scope='module')
def get_tensorsum(get_tensor):
    def get_tensorsum(ndim: int, size: int):
        tensors = [t[0] for t in get_tensor(ndim, number=size)]
        return TensorSum(tensors)
    return get_tensorsum
