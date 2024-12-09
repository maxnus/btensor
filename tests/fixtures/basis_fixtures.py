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

import pytest

from btensor import Basis


@pytest.fixture(params=[1, 5], scope='module')
def basis_size(request):
    return request.param


@pytest.fixture(params=[10], scope='module')
def basis_size_large(request):
    return request.param


@pytest.fixture(scope='module')
def basis(basis_size):
    return Basis(basis_size)


@pytest.fixture(scope='module')
def basis_large(basis_size_large):
    return Basis(basis_size_large)
