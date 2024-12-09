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

import pytest
import numpy as np


MAX_NDIM = 4


@pytest.fixture(params=range(1, MAX_NDIM+1), scope='module', ids=lambda x: f'ndim{x}')
def ndim(request):
    return request.param


@pytest.fixture(params=range(2, MAX_NDIM+1), scope='module', ids=lambda x: f'ndim{x}')
def ndim_atleast2(request):
    return request.param


@pytest.fixture(params=range(3, MAX_NDIM+1), scope='module', ids=lambda x: f'ndim{x}')
def ndim_atleast3(request):
    return request.param


@pytest.fixture(params=range(3), scope='module', ids=lambda x: f'seed{x}')
def rng(request):
    return np.random.default_rng(request.param)
