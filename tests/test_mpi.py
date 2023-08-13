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
from mpi4py import MPI

from helper import TestCase


MPI_SIZE = 4


@pytest.mark.mpi
class TestMPI(TestCase):

    @pytest.fixture
    def comm(self):
        return MPI.COMM_WORLD

    @pytest.fixture(params=range(MPI_SIZE))
    def root(self, request):
        return request.param

    def test_size(self, comm):
        assert comm.size == MPI_SIZE

    def test_bcast(self, comm, root):
        rng = np.random.default_rng(0)
        data = rng.random(100)
        data_bcast = comm.bcast(data, root=root)
        assert np.all(data == data_bcast)

    def test_Bcast(self, comm, root):
        rank = comm.Get_rank()
        rng = np.random.default_rng(0)
        data = rng.random(100)
        buffer = np.empty_like(data)
        comm.Bcast(data if rank == root else buffer, root=root)
        assert (rank == root) or np.all(buffer == data)
