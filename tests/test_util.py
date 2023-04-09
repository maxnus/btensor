import unittest
import itertools
import numpy as np
from basis_array import util
from testing import TestCase, powerset


class UtilTests(TestCase):

    def test_column_permutation_matrix(self):
        n = 10
        m = 5
        perm = np.random.permutation(range(n))[:m]
        p = util.ColumnPermutationMatrix(permutation=perm, size=n)
        pt = p.T
        self.assertAllclose(p.to_array().T, pt.to_array())
        self.assertAllclose(pt.to_array().dot(p.to_array()), np.identity(m))
        ppt = p.to_array().dot(pt.to_array())
        self.assertAllclose(ppt - np.diag(np.diag(ppt)), 0)
        nonzero = np.diag(ppt).nonzero()[0]
        self.assertAllclose(set(nonzero), set(perm.tolist()))

    def test_row_permutation_matrix(self):
        n = 5
        m = 10
        perm = np.random.permutation(range(m))[:n]
        p = util.RowPermutationMatrix(permutation=perm, size=m)
        pt = p.T
        self.assertAllclose(p.to_array().T, pt.to_array())
        self.assertAllclose(p.to_array().dot(pt.to_array()), np.identity(n))
        ptp = pt.to_array().dot(p.to_array())
        self.assertAllclose(ptp - np.diag(np.diag(ptp)), 0)
        nonzero = np.diag(ptp).nonzero()[0]
        self.assertAllclose(set(nonzero), set(perm.tolist()))

    def test_combine_permutation_matrices(self):
        n, m, k = 10, 8, 6
        # Column-column
        pc1 = np.random.permutation(range(n))[:m]
        pc2 = np.random.permutation(range(m))[:k]
        c1 = util.ColumnPermutationMatrix(permutation=pc1, size=n)  # n x m
        c2 = util.ColumnPermutationMatrix(permutation=pc2, size=m)  # m x k
        self.assertAllclose(util.chained_dot(c1, c2), np.dot(c1.to_array(), c2.to_array()))
        # Row-row
        pr1 = np.random.permutation(range(n))[:m]
        pr2 = np.random.permutation(range(m))[:k]
        r1 = util.RowPermutationMatrix(permutation=pr1, size=n)     # m x n
        r2 = util.RowPermutationMatrix(permutation=pr2, size=m)     # k x m
        self.assertAllclose(util.chained_dot(r2, r1), np.dot(r2.to_array(), r1.to_array()))
        # Column-row
        self.assertAllclose(util.chained_dot(c1, r1), np.dot(c1.to_array(), r1.to_array()))
        self.assertAllclose(util.chained_dot(c2, r2), np.dot(c2.to_array(), r2.to_array()))
        # Row-column
        self.assertAllclose(util.chained_dot(r1, c1), np.dot(r1.to_array(), c1.to_array()))
        self.assertAllclose(util.chained_dot(r2, c2), np.dot(r2.to_array(), c2.to_array()))




def generate_test_chained_dot(cls, atol=1e-10):
    n = 10
    i = util.IdentityMatrix(n)
    a = util.Matrix(np.random.rand(n, n))
    b = util.Matrix(np.random.rand(n, n))
    r = util.RowPermutationMatrix(permutation=np.random.permutation(n), size=n)
    #c = util.RowPermutationMatrix(permutation=np.random.permutation(n), size=n)
    c = util.ColumnPermutationMatrix(permutation=np.random.permutation(n), size=n)
    #r = util.ColumnPermutationMatrix(permutation=np.random.permutation(n), size=n)
    ainv = util.InverseMatrix(a)
    binv = util.InverseMatrix(b)

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
            self.assertAllclose(util.chained_dot(*args), ref, atol=atol, rtol=0)
        return test

    #matrices = {'x': None, 'i': i, 'a': a, 'b': b, 'ainv': ainv, 'binv': binv, 'c': c, 'r': r}
    matrices = {'x': None, 'i': i, 'a': a, 'ainv': ainv, 'c': c, 'r': r}
    matrices = [(k, v) for (k, v) in matrices.items()]

    for i, args in enumerate(powerset(matrices, include_empty=False)):
        for perm in itertools.permutations(args):
            mats = [p[1] for p in perm]
            name = '_'.join([p[0] for p in perm])
            funcname = 'test_chained_dot_%s' % name
            print("Adding test '%s'" % funcname)
            assert not hasattr(cls, funcname)
            setattr(cls, funcname, generate_test(*mats))


generate_test_chained_dot(UtilTests)


if __name__ == '__main__':
    unittest.main()
