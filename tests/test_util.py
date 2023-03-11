
import unittest
import itertools
import numpy as np

import basis_array as basis
from basis_array import util
from testing import TestCase, powerset


class UtilTests(TestCase):


    def test_chained_dot(self):
        n = 10
        i = util.IdentityMatrix(n)
        a = np.random.rand(n, n)
        b = np.random.rand(n, n)

        m = util.Matrix(np.random.rand(n, n))
        p = util.Matrix(np.random.rand(n, n))

        def assert_allclose(*args, atol):
            args_ref = [x for x in args if x is not None]
            args_ref = [getattr(x, 'values', x) for x in args_ref]
            if len(args_ref) == 0:
                ref = None
            elif len(args_ref) == 1:
                ref = args_ref[0]
            else:
                ref = np.linalg.multi_dot(args_ref)
            self.assertAllclose(util.chained_dot(*args), ref, atol=atol, rtol=0)

        for args in powerset([None, i, a, b, m, m.inv, p, p.inv], include_empty=False):
            for perm in itertools.permutations(args):
                assert_allclose(*perm, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
