import unittest
import itertools
import numpy as np
import scipy
import scipy.stats


def rand_orth_mat(n, ncol=None):
    m = scipy.stats.ortho_group.rvs(n)
    #return m
    if ncol is not None:
        m = m[:,:ncol]
    return m


def powerset(iterable, include_empty=True):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    start = 0 if include_empty else 1
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(start, len(s)+1))


class TestCase(unittest.TestCase):

    allclose_atol = 1e-13
    allclose_rtol = 0

    def assertAllclose(self, actual, desired, rtol=allclose_rtol, atol=allclose_atol, **kwargs):
        if actual is desired is None:
            return True
        # Compare multiple pairs of arrays:
        if isinstance(actual, (tuple, list)):
            for i in range(len(actual)):
                self.assertAllclose(actual[i], desired[i], rtol=rtol, atol=atol, **kwargs)
            return
        # Compare single pair of arrays:
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, **kwargs)
        #try:
        #    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, **kwargs)
        #except AssertionError as e:
        #    # Add higher precision output:
        #    message = e.args[0]
        #    args = e.args[1:]
        #    message += '\nHigh precision:\n x: %r\n y: %r' % (actual, desired)
        #    e.args = (message, *args)
        #    raise

    def setUp(self):
        np.random.seed(0)
