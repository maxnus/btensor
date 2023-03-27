import unittest
import itertools
import numpy as np
from basis_array import util
from testing import TestCase, powerset


class UtilTests(TestCase):
    pass


def generate_test_chained_dot(cls, atol=1e-10):
    n = 10
    i = util.IdentityMatrix(n)
    a = util.Matrix(np.random.rand(n, n))
    b = util.Matrix(np.random.rand(n, n))
    p = util.PermutationMatrix(order=[2,8,0,1,3,7,5,4,6,9], size=n)
    ainv = util.InverseMatrix(a)
    binv = util.InverseMatrix(b)

    def generate_test(*args):
        def test(self):
            args_ref = [x for x in args if x is not None]
            args_ref = [getattr(x, 'values', x) for x in args_ref]
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
