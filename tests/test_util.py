import unittest
import itertools
import numpy as np
from basis_array import util
from testing import TestCase, powerset


class UtilTests(TestCase):

    def test_combine_permutation_matrices(self):
        n, m, k = 10, 8, 6
        # Column-column
        pc1 = np.random.permutation(range(n))[:m]
        pc2 = np.random.permutation(range(m))[:k]
        c1 = util.ColumnPermutationMatrix(permutation=pc1, size=n)  # n x m
        c2 = util.ColumnPermutationMatrix(permutation=pc2, size=m)  # m x k
        self.assertAllclose(util.MatrixProduct((c1, c2)).evaluate(), np.dot(c1.to_array(), c2.to_array()))
        # Row-row
        pr1 = np.random.permutation(range(n))[:m]
        pr2 = np.random.permutation(range(m))[:k]
        r1 = util.RowPermutationMatrix(permutation=pr1, size=n)     # m x n
        r2 = util.RowPermutationMatrix(permutation=pr2, size=m)     # k x m
        self.assertAllclose(util.MatrixProduct((r2, r1)).evaluate(), np.dot(r2.to_array(), r1.to_array()))
        # Column-row
        self.assertAllclose(util.MatrixProduct((c1, r1)).evaluate(), np.dot(c1.to_array(), r1.to_array()))
        self.assertAllclose(util.MatrixProduct((c2, r2)).evaluate(), np.dot(c2.to_array(), r2.to_array()))
        # Row-column
        self.assertAllclose(util.MatrixProduct((r1, c1)).evaluate(), np.dot(r1.to_array(), c1.to_array()))
        self.assertAllclose(util.MatrixProduct((r2, c2)).evaluate(), np.dot(r2.to_array(), c2.to_array()))


def generate_test_permutation_matrix(cls, column=True):
    n = 10

    def generate_test(perm):
        m = len(np.arange(n)[perm])

        def test(self):
            nonlocal perm
            if column:
                p = util.ColumnPermutationMatrix(permutation=perm, size=n)
                pt = p.T
            else:
                pt = util.RowPermutationMatrix(permutation=perm, size=n)
                p = pt.T
            # Test transpose
            self.assertAllclose(p.to_array().T, pt.to_array())
            # Test p.T x p
            self.assertAllclose(pt.to_array().dot(p.to_array()), np.identity(m))
            # Test p x p.T
            ppt = p.to_array().dot(pt.to_array())
            self.assertAllclose(ppt - np.diag(np.diag(ppt)), 0)
            nonzero = np.diag(ppt).nonzero()[0]
            if isinstance(perm, slice):
                perm = np.arange(n)[perm]
            expected = set(np.asarray(perm).tolist())
            self.assertAllclose(set(nonzero), expected)
        return test

    # Test permutations:
    perms = [
        [], [0], [4],
        np.arange(n), np.arange(5), np.random.permutation(range(n))[:5],
        slice(None), slice(0, 5), slice(5, n), slice(None, None, 2), slice(None, None, -2), slice(3, 8),
    ]
    for m, perm in enumerate(perms):
        funcname = 'test_%s_permutation_matrix_%s_%d' % ('column' if column else 'row', type(perm).__name__, m)
        print("Adding test '%s'" % funcname)
        assert not hasattr(cls, funcname)
        setattr(cls, funcname, generate_test(perm))


def generate_test_chained_dot(cls, simplify=True, atol=1e-10):
    n = 10

    def generate_test(*args):
        def test(self):
            args_ref = [x for x in args if x is not None]
            args_ref = [(x.to_array() if hasattr(x, 'to_array') else x) for x in args_ref]
            if len(args_ref) == 0:
                ref = None
            elif len(args_ref) == 1:
                ref = args_ref[0]
            else:
                ref = np.linalg.multi_dot(args_ref)
            mpl = util.MatrixProduct(args)
            self.assertAllclose(mpl.evaluate(simplify=simplify), ref, atol=atol, rtol=0)
        return test

    a = util.GeneralMatrix(np.random.rand(n, n))
    matrices = {'x': None,
                'i': util.IdentityMatrix(n),
                'a': a,
                'b': a.inverse,
                'c': util.ColumnPermutationMatrix(permutation=np.random.permutation(n), size=n),
                'r': util.RowPermutationMatrix(permutation=np.random.permutation(n), size=n),
                }
    matrices = [(k, v) for (k, v) in matrices.items()]

    for i, args in enumerate(powerset(matrices, include_empty=False)):
        if len(args) == 1 and ('x', None) in args:
            continue
        for perm in itertools.permutations(args):
            mats = [p[1] for p in perm]
            name = ''.join([p[0] for p in perm])
            funcname = f'test_matrix_product_{"simplify" if simplify else ""}_{name}'
            print(f"Adding test {funcname}")
            assert not hasattr(cls, funcname)
            setattr(cls, funcname, generate_test(*mats))


generate_test_permutation_matrix(UtilTests, column=True)

generate_test_permutation_matrix(UtilTests, column=False)

generate_test_chained_dot(UtilTests, simplify=False)

generate_test_chained_dot(UtilTests, simplify=True)


if __name__ == '__main__':
    unittest.main()
