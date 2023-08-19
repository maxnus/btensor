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

from btensor import Basis, nobasis
from btensor.basistuple import BasisTuple


@pytest.fixture
def basistuple(shape_and_basis):
    shape, basis = shape_and_basis
    return BasisTuple(basis)


class TestBasistuple:

    @pytest.mark.parametrize('key', [0, slice(2), slice(1, 3), slice(None, None, -1)], ids=lambda x: str(x))
    def test_getitem(self, key, basistuple, shape_and_basis):
        expected = tuple(shape_and_basis[1])[key]
        assert basistuple[key] == expected

    @pytest.mark.parametrize('key', [(Basis(2),), (Basis(2), nobasis)], ids=lambda x: str(x))
    def test_valid_element(self, key: tuple):
        assert BasisTuple(key) == tuple(key)

    @pytest.mark.parametrize('key', [(1,), (slice(None),), (...,), (Basis(2), slice(None))], ids=lambda x: str(x))
    def test_raises_invalid_element(self, key: tuple):
        with pytest.raises(TypeError):
            BasisTuple(key)

    @pytest.mark.parametrize('basis', [slice(None), Ellipsis], ids=lambda x: str(x))
    def test_create_from_default_slice_ellipsis(self, basis):
        default = BasisTuple((Basis(1), Basis(2), nobasis, Basis(3)))
        result = BasisTuple.create_from_default(basis, default)
        assert result == default

    @pytest.mark.parametrize('basis', [Basis(5)], ids=lambda x: str(x))
    def test_create_from_default_1(self, basis):
        default = BasisTuple((Basis(1), Basis(2), nobasis, Basis(3)))
        result = BasisTuple.create_from_default(basis, default)
        assert result == BasisTuple((basis,)) + default[1:]

    @pytest.mark.parametrize('basis', [(slice(None), Basis(5)), (slice(None), nobasis)], ids=lambda x: str(x))
    def test_create_from_default_2(self, basis):
        default = BasisTuple((Basis(1), Basis(2), nobasis, Basis(3)))
        result = BasisTuple.create_from_default(basis, default)
        assert result == default[:1] + BasisTuple(basis[1:]) + default[2:]

    @pytest.mark.parametrize('basis', [(slice(None), slice(None), Basis(5)), (slice(None), slice(None), nobasis)],
                             ids=lambda x: str(x))
    def test_create_from_default_3(self, basis):
        default = BasisTuple((Basis(1), Basis(2), nobasis, Basis(3)))
        result = BasisTuple.create_from_default(basis, default)
        assert result == default[:2] + BasisTuple(basis[2:]) + default[3:]
