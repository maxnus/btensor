import pytest

from helper import TestCase
from btensor import decomp


class TestDecomp(TestCase):

    @pytest.mark.parametrize('dim', [3, 4, 5, 6])
    def test_hosvd(self, get_tensor, dim):
        tensor, nparray = get_tensor(ndim=dim)
        hosvd = decomp.hosvd(tensor)
        delta = (hosvd - tensor).to_numpy()
        self.assert_allclose(delta, 0)
