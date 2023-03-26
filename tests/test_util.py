
import unittest
import itertools
import numpy as np

import basis_array as basis
from basis_array import util
from testing import TestCase, powerset


class UtilTests(TestCase):
    pass


def generate_test_chained_dot(cls, atol=1e-10):
    n = 10
    i = util.IdentityMatrix(n)
    a = util.Matrix(np.random.rand(n, n))
    b = util.Matrix(np.random.rand(n, n))
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

    for i, args in enumerate(powerset([None, i, a, b, ainv, binv], include_empty=False)):
        for j, perm in enumerate(itertools.permutations(args)):
            funcname = 'test_chained_dot_set%d_perm%d' % (i, j)
            print("Adding test '%s'" % funcname)
            setattr(cls, funcname, generate_test(*perm))


generate_test_chained_dot(UtilTests)


if __name__ == '__main__':
    unittest.main()
