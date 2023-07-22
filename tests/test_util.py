import itertools
import pytest
import numpy as np

from btensor import util
from helper import TestCase, powerset


class TestPermutationMatrix(TestCase):

    def test_combine_permutation_matrices(self):
        n, m, k = 10, 8, 6
        # Column-column
        pc1 = np.random.permutation(range(n))[:m]
        pc2 = np.random.permutation(range(m))[:k]
        c1 = util.ColumnPermutationMatrix(permutation=pc1, size=n)  # n x m
        c2 = util.ColumnPermutationMatrix(permutation=pc2, size=m)  # m x k
        self.assert_allclose(util.MatrixProductList((c1, c2)).evaluate(), np.dot(c1.to_numpy(), c2.to_numpy()))
        # Row-row
        pr1 = np.random.permutation(range(n))[:m]
        pr2 = np.random.permutation(range(m))[:k]
        r1 = util.RowPermutationMatrix(permutation=pr1, size=n)     # m x n
        r2 = util.RowPermutationMatrix(permutation=pr2, size=m)     # k x m
        self.assert_allclose(util.MatrixProductList((r2, r1)).evaluate(), np.dot(r2.to_numpy(), r1.to_numpy()))
        # Column-row
        self.assert_allclose(util.MatrixProductList((c1, r1)).evaluate(), np.dot(c1.to_numpy(), r1.to_numpy()))
        self.assert_allclose(util.MatrixProductList((c2, r2)).evaluate(), np.dot(c2.to_numpy(), r2.to_numpy()))
        # Row-column
        self.assert_allclose(util.MatrixProductList((r1, c1)).evaluate(), np.dot(r1.to_numpy(), c1.to_numpy()))
        self.assert_allclose(util.MatrixProductList((r2, c2)).evaluate(), np.dot(r2.to_numpy(), c2.to_numpy()))

    @staticmethod
    def get_permutation_matrix_input():
        np.random.seed(0)
        size = 5
        return [
            [], [0], [4],
            np.arange(size), np.arange(5), np.random.permutation(range(size))[:5],
            slice(None), slice(0, 5), slice(5, size), slice(None, None, 2), slice(None, None, -2), slice(3, 8)
        ]

    @pytest.fixture(params=get_permutation_matrix_input.__func__(), ids=str)
    def permutation_matrix_input(self, request):
        return request.param

    @pytest.fixture(params=['row', 'column'])
    def permtuation_matrix_type(self, request):
        return request.param

    def test_permutation_matrix(self, permutation_matrix_input, permtuation_matrix_type):
        perm = permutation_matrix_input
        n = 5
        m = len(np.arange(n)[perm])
        if permtuation_matrix_type == 'column':
            p = util.ColumnPermutationMatrix(permutation=perm, size=n)
            pt = p.T
        elif permtuation_matrix_type == 'row':
            pt = util.RowPermutationMatrix(permutation=perm, size=n)
            p = pt.T
        # Test transpose
        self.assert_allclose(p.to_numpy().T, pt.to_numpy())
        # Test p.T x p
        self.assert_allclose(pt.to_numpy().dot(p.to_numpy()), np.identity(m))
        # Test p x p.T
        ppt = p.to_numpy().dot(pt.to_numpy())
        self.assert_allclose(ppt - np.diag(np.diag(ppt)), 0)
        nonzero = np.diag(ppt).nonzero()[0]
        if isinstance(perm, slice):
            perm = np.arange(n)[perm]
        expected = set(np.asarray(perm).tolist())
        self.assert_allclose(set(nonzero), expected)


class TestMatrixProduct(TestCase):

    @pytest.fixture(params=['', 'simplify'])
    def matrix_product_simplify(self, request):
        return request.param

    @staticmethod
    def get_matrices():
        n = 10
        a = util.GeneralMatrix(np.random.rand(n, n))
        matrices = {'i': util.IdentityMatrix(n),
                    'a': a,
                    'b': a.inverse,
                    'c': util.ColumnPermutationMatrix(permutation=np.random.permutation(n), size=n),
                    'r': util.RowPermutationMatrix(permutation=np.random.permutation(n), size=n),
                    }
        matrices = [(k, v) for (k, v) in matrices.items()]
        for i, args in enumerate(powerset(matrices, include_empty=False)):
            for perm in itertools.permutations(args):
                name = ''.join([p[0] for p in perm])
                mats = [p[1] for p in perm]
                yield name, mats

    @pytest.fixture(params=get_matrices.__func__(), ids=lambda x: x[0])
    def matrices(self, request):
        return request.param

    def test_chained_dot(self, matrices, matrix_product_simplify, atol=1e-10):
        matrices = matrices[1]
        args_ref = [x for x in matrices if x is not None]
        args_ref = [(x.to_numpy() if hasattr(x, 'to_numpy') else x) for x in args_ref]
        if len(args_ref) == 0:
            ref = None
        elif len(args_ref) == 1:
            ref = args_ref[0]
        else:
            ref = np.linalg.multi_dot(args_ref)
        mpl = util.MatrixProductList(matrices)
        result = mpl.evaluate(simplify=bool(matrix_product_simplify))
        self.assert_allclose(mpl.evaluate(simplify=bool(matrix_product_simplify)), ref, atol=atol, rtol=0)
