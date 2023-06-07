import numpy as np

import btensor as basis
from helper import TestCase, rand_orth_mat


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

        cls.rootbasis_a = ba = basis.Basis(cls.size_a, metric=cls.metric_a)
        cls.rootbasis_b = bb = basis.Basis(cls.size_b, metric=cls.metric_b)

        # Subbasis

        def make_subbasis(b, permutation=False):
            subbasis = []
            trafos = []
            parent = b
            while parent.size > 1:
                if permutation:
                    t = np.random.permutation(range(parent.size))[:parent.size-1]
                    trafos.append(np.identity(parent.size)[:,t])
                else:
                    t = rand_orth_mat(parent.size, parent.size-1)
                    trafos.append(t)
                b = basis.Basis(t, parent=parent)
                subbasis.append(b)
                parent = b
            return subbasis, trafos

        cls.subbasis_a, cls.trafos_a = make_subbasis(cls.rootbasis_a)
        cls.subbasis_b, cls.trafos_b = make_subbasis(cls.rootbasis_b, permutation=True)

        cls.basis_a = [cls.rootbasis_a, *cls.subbasis_a]
        cls.basis_b = [cls.rootbasis_b, *cls.subbasis_b]

    def test_is_root(self):
        assert (self.rootbasis_a.is_root())
        assert (self.rootbasis_b.is_root())
        for b in self.subbasis_a:
            assert not (b.is_root())
        for b in self.subbasis_b:
            assert not (b.is_root())

    def test_size(self):
        assert (self.rootbasis_a.size == self.size_a)
        assert (self.rootbasis_b.size == self.size_b)
        for i, b in enumerate(self.subbasis_a):
            assert (b.size == self.rootbasis_a.size - (i + 1))

    #def test_hash(self, get_rootbasis_subbasis, subbasis_type):
    #    rootsize = subsize = 10
    #    rootbasis, (subbasis, subarg) = get_rootbasis_subbasis(rootsize, subsize, subbasis_type)
    #    #print(hash(rootbasis))
    #    print(hash(subbasis))

    def test_pos_neg_invert(self):
        b = self.basis_a[0]
        d = b.dual()
        assert (-b == d)
        assert (~b == d)
        assert (+d == b)
        assert (~d == b)

    def test_len_and_ordering(self):
        for i, b1 in enumerate(self.basis_a):
            for j, b2 in enumerate(self.basis_a):
                if i == j:
                    assert (len(b1) == len(b2))
                    assert (len(b1) >= len(b2))
                    assert (len(b1) <= len(b2))
                    assert not (len(b1) > len(b2))
                    assert not (len(b1) < len(b2))
                    assert not (len(b1) != len(b2))
                    assert (len(b2) == len(b1))
                    assert (len(b2) >= len(b1))
                    assert (len(b2) <= len(b1))
                    assert not (len(b2) > len(b1))
                    assert not (len(b2) < len(b1))
                    assert not (len(b2) != len(b1))
                elif i > j:
                    assert not (len(b1) == len(b2))
                    assert not (len(b1) >= len(b2))
                    assert (len(b1) <= len(b2))
                    assert not (len(b1) > len(b2))
                    assert (len(b1) < len(b2))
                    assert (len(b1) != len(b2))
                    assert not (len(b2) == len(b1))
                    assert (len(b2) >= len(b1))
                    assert not (len(b2) <= len(b1))
                    assert (len(b2) > len(b1))
                    assert not (len(b2) < len(b1))
                    assert (len(b2) != len(b1))
                else:
                    assert not (len(b1) == len(b2))
                    assert (len(b1) >= len(b2))
                    assert not (len(b1) <= len(b2))
                    assert (len(b1) > len(b2))
                    assert not (len(b1) < len(b2))
                    assert (len(b1) != len(b2))
                    assert not (len(b2) == len(b1))
                    assert not (len(b2) >= len(b1))
                    assert (len(b2) <= len(b1))
                    assert not (len(b2) > len(b1))
                    assert (len(b2) < len(b1))
                    assert (len(b2) != len(b1))

    def test_space_same_root(self):
        for bas in (self.basis_a, self.basis_b):
            for i, b1 in enumerate(bas):
                for j, b2 in enumerate(bas):
                    # space(b1) == space(b2)
                    if i == j:
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
                    # space(b1) < space(b2)
                    elif i > j:
                        assert not (b1.space == b2.space)
                        assert (b1.space != b2.space)
                        assert not (b1.space >= b2.space)
                        assert (b1.space <= b2.space)
                        assert not (b1.space > b2.space)
                        assert (b1.space < b2.space)
                        assert not (b2.space == b1.space)
                        assert (b2.space != b1.space)
                        assert (b2.space >= b1.space)
                        assert not (b2.space <= b1.space)
                        assert (b2.space > b1.space)
                        assert not (b2.space < b1.space)
                    # space(b1) > space(b2)
                    else:
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

    def test_same_space(self):
        for bas in (self.basis_a, self.basis_b):
            for b in bas:
                b1 = b
                b2 = basis.Basis(rand_orth_mat(b.size), parent=b)
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
                b1 = basis.Basis(rand_orth_mat(b.size), parent=b)
                b2 = basis.Basis(rand_orth_mat(b.size), parent=b)
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
                b1 = basis.Basis(r, parent=b)
                b2 = basis.Basis(r, parent=b)
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
                b1 = basis.Basis(r1, parent=b)
                b2 = basis.Basis(r2, parent=b)
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
                b1 = basis.Basis(r1, parent=b)
                b2 = basis.Basis(r2, parent=b)
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
                b1 = basis.Basis(r1, parent=b)
                b2 = basis.Basis(r2, parent=b)
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
        b1 = b.make_basis([0, 1])
        b2 = b.make_basis([2, 3, 4])
        assert (b1.space | b2.space)
        b = self.basis_b[0]
        b1 = b.make_basis([0, 1])
        b2 = b.make_basis([2, 3, 4])
        assert (b1.space | b2.space)
