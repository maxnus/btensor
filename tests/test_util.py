import unittest
import itertools
import numpy as np
from basis_array import util
from testing import TestCase, powerset


class UtilTests(TestCase):

    def test_permutation_matrix(self):
        n = 10
        m = 5
        d = np.random.random((n, m,))
        perm = np.random.permutation(range(n))[:m]
        p = util.PermutationMatrix(permutation=perm, size=n)
        pt = p.T
        self.assertAllclose(p.to_array().T, pt.to_array())
        self.assertAllclose(pt.to_array().dot(p.to_array()), np.identity(m))
        ppt = p.to_array().dot(pt.to_array())
        self.assertAllclose(ppt - np.diag(np.diag(ppt)), 0)
        nonzero = np.diag(ppt).nonzero()[0]
        self.assertAllclose(set(nonzero), set(perm.tolist()))

    def test_permutation_matrix_axis0(self):
        n = 5
        m = 10
        d = np.random.random((n, m))
        perm = np.random.permutation(range(m))[:n]
        p = util.PermutationMatrix(permutation=perm, size=m, axis=0)
        pt = p.T
        self.assertAllclose(p.to_array().T, pt.to_array())
        self.assertAllclose(p.to_array().dot(pt.to_array()), np.identity(n))
        ptp = pt.to_array().dot(p.to_array())
        self.assertAllclose(ptp - np.diag(np.diag(ptp)), 0)
        nonzero = np.diag(ptp).nonzero()[0]
        self.assertAllclose(set(nonzero), set(perm.tolist()))


def generate_test_chained_dot(cls, atol=1e-10):
    n = 10
    i = util.IdentityMatrix(n)
    a = util.Matrix(np.random.rand(n, n))
    b = util.Matrix(np.random.rand(n, n))
    p = util.PermutationMatrix(permutation=[2, 8, 0, 1, 3, 7, 5, 4, 6, 9], size=n)
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

    matrices = {'x': None, 'i': i, 'a': a, 'b': b, 'ainv': ainv, 'binv': binv, 'p': p}
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
