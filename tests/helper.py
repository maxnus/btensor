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

from __future__ import annotations
import itertools
from contextlib import contextmanager
from numbers import Number
from time import perf_counter
from typing import *

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
    """powerset([1, 2, 3]) --> [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]"""
    s = list(iterable)
    start = 0 if include_empty else 1
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(start, len(s)+1))


class TestTimings:

    def __init__(self, test=None):
        self._test = test
        self._timings = {}

    def __len__(self) -> int:
        return len(self._timings)

    @contextmanager
    def __call__(self, timer_name: str) -> None:
        if timer_name not in self._timings:
            self._timings[timer_name] = (0, 0.0)
        start = perf_counter()
        yield
        self._timings[timer_name] = (self._timings[timer_name][0] + 1,
                                     self._timings[timer_name][1] + (perf_counter() - start))
        return

    def print_report(self) -> None:
        header = "Timings"
        if self._test is not None:
            header = f"{header} {type(self._test).__name__}"
        print()
        print(header)
        print(len(header)*'=')
        key_width = max(map(len, self._timings.keys()))
        for key, (count, time) in self._timings.items():
            print(f"{key + ':':{key_width + 1}} {count:4d} calls  in  {time:.3f} s")


class TestCase:

    allclose_atol = 1e-14
    allclose_rtol = 1e-10

    @pytest.fixture(scope='module')
    def timings(self) -> TestTimings:
        return TestTimings(self)

    @pytest.fixture(scope='module', autouse=True)
    def report_timings(self, timings) -> None:
        yield
        if len(timings):
            timings.print_report()

    def assert_allclose(self,
                        actual: np.ndarray | btensor.Tensor | Collection,
                        desired: np.ndarray | btensor.Tensor | Collection,
                        rtol: float | None = None, atol: float | None = None, **kwargs: Any) -> None:
        if rtol is None:
            rtol = self.allclose_rtol
        if atol is None:
            atol = self.allclose_atol
        if actual is desired is None:
            return
        # TODO: Floats in set
        if isinstance(actual, set) and isinstance(desired, set):
            assert actual == desired
            return
        # Compare multiple pairs of arrays:
        if isinstance(actual, (tuple, list)):
            for i in range(len(actual)):
                self.assert_allclose(actual[i], desired[i], rtol=rtol, atol=atol, **kwargs)
            return

        def to_array(obj):
            if isinstance(obj, (Number, np.ndarray)):
                return obj
            if isinstance(obj, btensor.Tensor):
                return obj.to_numpy()
            raise TypeError(f"unknown type: {type(obj)}")

        actual = to_array(actual)
        desired = to_array(desired)
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, **kwargs)


class TestCase1Array(TestCase):

    @pytest.fixture(autouse=True)
    def init_tensors(self, array):
        self.array, self.data = array
