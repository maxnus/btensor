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

import numpy as np
import pytest

import btensor
from helper import TestCase, rand_orth_mat
from conftest import get_random_subbasis_definition


@pytest.fixture
def size_a():
    return 10

@pytest.fixture(params=[None, 0, 0.1, 1.0, 3.0],
                ids=['Orth', 'ZeroNonOrth', 'SmallNonOrth', 'MediumNonOrth', 'LargeNonOrth'])
def metric_nonorth_factor(request):
    return request.param

@pytest.fixture
def metric_a(metric_nonorth_factor, size_a, rng):
    if metric_nonorth_factor is None:
        return None
    noise = metric_nonorth_factor * (rng.random((size_a, size_a)) - 0.5)
    noise = np.dot(noise, noise.T)
    #np.fill_diagonal(noise, 0)
    return np.identity(size_a) + noise


@pytest.fixture
def basis_a0(size_a, metric_a):
    return btensor.Basis(size_a, metric=metric_a)


@pytest.fixture
def basis_a1(basis_a0, subbasis_type, rng):
    d = get_random_subbasis_definition(len(basis_a0), len(basis_a0) - 2, subbasis_type, rng=rng)
    return basis_a0.make_subbasis(d)


@pytest.fixture
def basis_a2(basis_a1, subbasis_type, rng):
    d = get_random_subbasis_definition(len(basis_a1), len(basis_a1) - 2, subbasis_type, rng=rng)
    return basis_a1.make_subbasis(d)


@pytest.fixture
def basis_a3(basis_a2, subbasis_type, rng):
    d = get_random_subbasis_definition(len(basis_a2), len(basis_a2) - 2, subbasis_type, rng=rng)
    return basis_a2.make_subbasis(d)


@pytest.fixture
def basis_as(basis_a0, basis_a1, basis_a2, basis_a3):
    return [basis_a0, basis_a1, basis_a2, basis_a3]


@pytest.fixture(params=range(4))
def basis_a(request, basis_as):
    return basis_as[request.param]


@pytest.fixture(params=range(4))
def basis_a_two(request, basis_as):
    return basis_as[request.param]


class TestBasisNew(TestCase):

    def test_is_root(self, basis_a0):
        assert basis_a0.is_root()

    def test_is_not_root(self, basis_a1):
        assert not basis_a1.is_root()

    def test_length_same_space(self, basis_a):
        assert len(basis_a) == len(basis_a)
        assert (len(basis_a) >= len(basis_a))
        assert (len(basis_a) <= len(basis_a))
        assert not (len(basis_a) > len(basis_a))
        assert not (len(basis_a) < len(basis_a))
        assert not (len(basis_a) != len(basis_a))
        assert (len(basis_a) == len(basis_a))
        assert (len(basis_a) >= len(basis_a))
        assert (len(basis_a) <= len(basis_a))
        assert not (len(basis_a) > len(basis_a))
        assert not (len(basis_a) < len(basis_a))
        assert not (len(basis_a) != len(basis_a))

    @pytest.mark.parametrize('ij', list(zip(*np.tril_indices(4, -1))), ids=str)
    def test_length_super_space(self, ij, basis_as):
        i, j = ij
        bi = basis_as[i]
        bj = basis_as[j]
        assert not (len(bi) == len(bj))
        assert not (len(bi) >= len(bj))
        assert (len(bi) <= len(bj))
        assert not (len(bi) > len(bj))
        assert (len(bi) < len(bj))
        assert (len(bi) != len(bj))
        assert not (len(bj) == len(bi))
        assert (len(bj) >= len(bi))
        assert not (len(bj) <= len(bi))
        assert (len(bj) > len(bi))
        assert not (len(bj) < len(bi))
        assert (len(bj) != len(bi))

    @pytest.mark.parametrize('ij', list(zip(*np.triu_indices(4, 1))), ids=str)
    def test_length_sub_space(self, ij, basis_as):
        i, j = ij
        bi = basis_as[i]
        bj = basis_as[j]
        assert not (len(bi) == len(bj))
        assert (len(bi) >= len(bj))
        assert not (len(bi) <= len(bj))
        assert (len(bi) > len(bj))
        assert not (len(bi) < len(bj))
        assert (len(bi) != len(bj))
        assert not (len(bj) == len(bi))
        assert not (len(bj) >= len(bi))
        assert (len(bj) <= len(bi))
        assert not (len(bj) > len(bi))
        assert (len(bj) < len(bi))
        assert (len(bj) != len(bi))

    def test_same_space(self, basis_a):
        assert (basis_a.space == basis_a.space)
        assert not (basis_a.space != basis_a.space)
        assert (basis_a.space >= basis_a.space)
        assert (basis_a.space <= basis_a.space)
        assert not (basis_a.space > basis_a.space)
        assert not (basis_a.space < basis_a.space)
        assert (basis_a.space == basis_a.space)
        assert not (basis_a.space != basis_a.space)
        assert (basis_a.space >= basis_a.space)
        assert (basis_a.space <= basis_a.space)
        assert not (basis_a.space > basis_a.space)
        assert not (basis_a.space < basis_a.space)

    @pytest.mark.parametrize('ij', list(zip(*np.tril_indices(4, -1))), ids=str)
    def test_super_space(self, ij, basis_as):
        i, j = ij
        bi = basis_as[i]
        bj = basis_as[j]
        # space(b1) < space(b2)
        assert i > j
        assert not (bi.space == bj.space)
        assert (bi.space != bj.space)
        assert not (bi.space >= bj.space)
        assert (bi.space <= bj.space)
        assert not (bi.space > bj.space)
        assert (bi.space < bj.space)
        assert not (bj.space == bi.space)
        assert (bj.space != bi.space)
        assert (bj.space >= bi.space)
        assert not (bj.space <= bi.space)
        assert (bj.space > bi.space)
        assert not (bj.space < bi.space)

    @pytest.mark.parametrize('ij', list(zip(*np.triu_indices(4, 1))), ids=str)
    def test_sub_space(self, ij, basis_as):
        i, j = ij
        bi = basis_as[i]
        bj = basis_as[j]
        # space(b1) > space(b2)
        assert i < j
        assert not (bi.space == bj.space)
        assert (bi.space != bj.space)
        assert (bi.space >= bj.space)
        assert not (bi.space <= bj.space)
        assert (bi.space > bj.space)
        assert not (bi.space < bj.space)
        assert not (bj.space == bi.space)
        assert (bj.space != bi.space)
        assert not (bj.space >= bi.space)
        assert (bj.space <= bi.space)
        assert not (bj.space > bi.space)
        assert (bj.space < bi.space)

    def test_union(self, basis_a0, basis_a2, subbasis_type, rng):
        bi = basis_a2
        d = get_random_subbasis_definition(len(basis_a0), len(basis_a2), subbasis_type, rng=rng)
        bj = basis_a0.make_subbasis(d)
        bij = bi.make_union_basis(bj)
        assert len(bij) >= max(len(bi), len(bj))
        assert len(bij) <= len(bi.get_common_parent(bj))
        assert bi.space <= bij.space
        assert bj.space <= bij.space
        t = btensor.Tensor(rng.random((len(bi), len(bi))), basis=(bi, bi))
        t2 = t.cob[bij, bij][bi, bi]
        self.assert_allclose(t - t2, 0)
        t = btensor.Tensor(rng.random((len(bj), len(bj))), basis=(bj, bj))
        t2 = t.cob[bij, bij][bj, bj]
        self.assert_allclose(t - t2, 0)

    def test_intersect(self, basis_a0, basis_a2, subbasis_type, rng, metric_nonorth_factor):
        bi = basis_a2
        d = get_random_subbasis_definition(len(basis_a0), len(basis_a2), subbasis_type, rng=rng)
        bj = basis_a0.make_subbasis(d)
        bij = bi.make_intersect_basis(bj)
        assert len(bij) <= min(len(bi), len(bj))
        assert bi.space >= bij.space
        assert bj.space >= bij.space
        t = btensor.Tensor(rng.random((len(bij), len(bij))), basis=(bij, bij))
        t2 = t.cob[bi, bi][bij, bij]
        self.assert_allclose(t - t2, 0)
        t = btensor.Tensor(rng.random((len(bij), len(bij))), basis=(bij, bij))
        t2 = t.cob[bj, bj][bij, bij]
        self.assert_allclose(t - t2, 0)


class TestBasis(TestCase):

    non_orth_noise = 0.1

    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        cls.size_a, cls.size_b = 5, 6

        def metric(n):
            noise = cls.non_orth_noise * (np.random.random((n, n))-0.5)
            return np.identity(n) + noise + noise.T

        cls.metric_a = metric(cls.size_a)
        cls.metric_b = metric(cls.size_b)

        cls.rootbasis_a = ba = btensor.Basis(cls.size_a, metric=cls.metric_a)
        cls.rootbasis_b = bb = btensor.Basis(cls.size_b, metric=cls.metric_b)

        # Subbasis

        def make_subbasis(b, permutation=False):
            subbasis = []
            trafos = []
            parent = b
            while parent.size > 1:
                if permutation:
                    t = np.random.permutation(range(parent.size))[:parent.size-1]
                    trafos.append(np.identity(parent.size)[:, t])
                else:
                    t = rand_orth_mat(parent.size, parent.size-1)
                    trafos.append(t)
                b = btensor.Basis(t, parent=parent)
                subbasis.append(b)
                parent = b
            return subbasis, trafos

        cls.subbasis_a, cls.trafos_a = make_subbasis(cls.rootbasis_a)
        cls.subbasis_b, cls.trafos_b = make_subbasis(cls.rootbasis_b, permutation=True)

        cls.basis_a = [cls.rootbasis_a, *cls.subbasis_a]
        cls.basis_b = [cls.rootbasis_b, *cls.subbasis_b]

    def test_same_space(self):
        for bas in (self.basis_a, self.basis_b):
            for b in bas:
                b1 = b
                b2 = btensor.Basis(rand_orth_mat(b.size), parent=b)
                assert (b1.space == b2.space)
                assert not (b1.space != b2.space)
                assert (b1.space >= b2.space)
                assert (b1.space <= b2.space)
                assert not (b1.space > b2.space)
                assert not (b1.space < b2.space)
                assert (b2.space == b1.space)
                assert not (b2.space != b1.space)
                assert (b2.space >= b1.space)
                assert (b2.space <= b1.space)
                assert not (b2.space > b1.space)
                assert not (b2.space < b1.space)

    def test_same_space_2(self):
        for bas in (self.basis_a, self.basis_b):
            for i, b in enumerate(bas):
                b1 = btensor.Basis(rand_orth_mat(b.size), parent=b)
                b2 = btensor.Basis(rand_orth_mat(b.size), parent=b)
                assert (b1.space == b2.space)
                assert not (b1.space != b2.space)
                assert (b1.space >= b2.space)
                assert (b1.space <= b2.space)
                assert not (b1.space > b2.space)
                assert not (b1.space < b2.space)
                assert (b2.space == b1.space)
                assert not (b2.space != b1.space)
                assert (b2.space >= b1.space)
                assert (b2.space <= b1.space)
                assert not (b2.space > b1.space)
                assert not (b2.space < b1.space)

    def test_svd_same_space(self):
        for bas in (self.basis_a, self.basis_b):
            for i, b in enumerate(bas[:-1]):
                r = rand_orth_mat(b.size, b.size-1)
                b1 = btensor.Basis(r, parent=b)
                b2 = btensor.Basis(r, parent=b)
                assert (b1.space == b2.space)
                assert not (b1.space != b2.space)
                assert (b1.space >= b2.space)
                assert (b1.space <= b2.space)
                assert not (b1.space > b2.space)
                assert not (b1.space < b2.space)
                assert (b2.space == b1.space)
                assert not (b2.space != b1.space)
                assert (b2.space >= b1.space)
                assert (b2.space <= b1.space)
                assert not (b2.space > b1.space)
                assert not (b2.space < b1.space)

    def test_svd_different_space_same_size(self):
        for bas in (self.basis_a, self.basis_b):
            for i, b in enumerate(bas[:-1]):
                r1 = rand_orth_mat(b.size, b.size - 1)
                r2 = rand_orth_mat(b.size, b.size - 1)
                b1 = btensor.Basis(r1, parent=b)
                b2 = btensor.Basis(r2, parent=b)
                assert not (b1.space == b2.space)
                assert (b1.space != b2.space)
                assert not (b1.space >= b2.space)
                assert not (b1.space <= b2.space)
                assert not (b1.space > b2.space)
                assert not (b1.space < b2.space)
                assert not (b2.space == b1.space)
                assert (b2.space != b1.space)
                assert not (b2.space >= b1.space)
                assert not (b2.space <= b1.space)
                assert not (b2.space > b1.space)
                assert not (b2.space < b1.space)

    def test_svd_different_space_different_size(self):
        for bas in (self.basis_a, self.basis_b):
            for i, b in enumerate(bas[:-2]):
                r1 = rand_orth_mat(b.size, b.size - 1)
                r2 = rand_orth_mat(b.size, b.size - 2)
                b1 = btensor.Basis(r1, parent=b)
                b2 = btensor.Basis(r2, parent=b)
                assert not (b1.space == b2.space)
                assert (b1.space != b2.space)
                assert not (b1.space >= b2.space)
                assert not (b1.space <= b2.space)
                assert not (b1.space > b2.space)
                assert not (b1.space < b2.space)
                assert not (b2.space == b1.space)
                assert (b2.space != b1.space)
                assert not (b2.space >= b1.space)
                assert not (b2.space <= b1.space)
                assert not (b2.space > b1.space)
                assert not (b2.space < b1.space)

    def test_svd_subspace(self):
        for bas in (self.basis_a, self.basis_b):
            for i, b in enumerate(bas[:-2]):
                r1 = rand_orth_mat(b.size, b.size - 1)
                r2 = r1[:, :-1]
                # b2 is subspace of b1
                b1 = btensor.Basis(r1, parent=b)
                b2 = btensor.Basis(r2, parent=b)
                assert not (b1.space == b2.space)
                assert (b1.space != b2.space)
                assert (b1.space >= b2.space)
                assert not (b1.space <= b2.space)
                assert (b1.space > b2.space)
                assert not (b1.space < b2.space)
                assert not (b2.space == b1.space)
                assert (b2.space != b1.space)
                assert not (b2.space >= b1.space)
                assert (b2.space <= b1.space)
                assert not (b2.space > b1.space)
                assert (b2.space < b1.space)

    def test_space_different_root(self):
        for b1 in self.basis_a:
            for b2 in self.basis_b:
                assert not (b1.space == b2.space)
                assert (b1.space != b2.space)
                assert not (b1.space >= b2.space)
                assert not (b1.space <= b2.space)
                assert not (b1.space > b2.space)
                assert not (b1.space < b2.space)
                assert not (b2.space == b1.space)
                assert (b2.space != b1.space)
                assert not (b2.space >= b1.space)
                assert not (b2.space <= b1.space)
                assert not (b2.space > b1.space)
                assert not (b2.space < b1.space)

    def test_space_orthogonal(self):
        for b1 in self.basis_a:
            assert not (b1.space | b1.space)
        for i, b1 in enumerate(self.basis_a):
            for j, b2 in enumerate(self.basis_a):
                assert not (b1.space | b2.space)
        for i, b1 in enumerate(self.basis_a):
            for j, b2 in enumerate(self.basis_b):
                assert (b1.space | b2.space)

    def test_matrices_for_coeff_in_basis(self):

        def test(basis1, basis2, expected):
            mats = basis1.coeff_in_basis(basis2)
            self.assert_allclose(mats.evaluate(), expected)

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
                    if len(mats) == 1:
                        expected = mats[0]
                    else:
                        expected = np.linalg.multi_dot(mats)
                test(b1, b2, expected)


class TestBasisOrthogonal(TestBasis):

    non_orth_noise = 0

    def test_space_orthogonal_disjoint(self):
        b = self.basis_a[0]
        b1 = b.make_subbasis([0, 1])
        b2 = b.make_subbasis([2, 3, 4])
        assert (b1.space | b2.space)
        b = self.basis_b[0]
        b1 = b.make_subbasis([0, 1])
        b2 = b.make_subbasis([2, 3, 4])
        assert (b1.space | b2.space)
