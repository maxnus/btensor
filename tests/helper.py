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

import itertools

import pytest
import numpy as np
import scipy
import scipy.stats

import btensor


def rand_orth_mat(n, ncol=None):
    if n == 1:
        return np.asarray([[1.0]])[:, :ncol]
    m = scipy.stats.ortho_group.rvs(n)
    if ncol is not None:
        m = m[:, :ncol]
    return m


def powerset(iterable, include_empty=True):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    start = 0 if include_empty else 1
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(start, len(s)+1))


class TestCase:

    allclose_atol = 1e-14
    allclose_rtol = 1e-12

    def assert_allclose(self, actual, desired, rtol=allclose_rtol, atol=allclose_atol, **kwargs):
        if actual is desired is None:
            return True
        # TODO: Floats in set
        if isinstance(actual, set) and isinstance(desired, set):
            return actual == desired
        # Compare multiple pairs of arrays:
        if isinstance(actual, (tuple, list)):
            for i in range(len(actual)):
                self.assert_allclose(actual[i], desired[i], rtol=rtol, atol=atol, **kwargs)
            return
        # Tensor does not have __array_interface__:
        if isinstance(actual, btensor.Tensor) and not hasattr(actual, '__array_interface__'):
            actual = actual._data
        if isinstance(desired, btensor.Tensor) and not hasattr(desired, '__array_interface__'):
            desired = desired._data

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


class TestCase1Array(TestCase):

    @pytest.fixture(autouse=True)
    def init_tensors(self, array):
        self.array, self.data = array
