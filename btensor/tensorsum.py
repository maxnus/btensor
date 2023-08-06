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
from typing import Union

from btensor import Tensor


class TensorSum:

    def __init__(self, tensors: list[Tensor]):
        self._tensors = []
        for tensor in tensors:
            self.add_tensor(tensor)

    @property
    def tensors(self):
        return self._tensors

    def add_tensor(self, tensor: Tensor):
        if len(self):
            if self.tensors[0].basis.get_root_basistuple() != tensor.basis.get_root_basistuple():
                raise ValueError
        self.tensors.append(tensor)

    def __len__(self):
        return len(self.tensors)

    def evaluate(self):
        return sum(self.tensors)

    def dot(self, other: Union[Tensor, 'TensorSum']) -> 'TensorSum':
        out = TensorSum([])
        for tensor in self.tensors:
            if isinstance(other, Tensor):
                out.add_tensor(tensor.dot(other))
            else:
                for tensor2 in other.tensors:
                    out.add_tensor(tensor.dot(tensor2))
        return out


#def test:
#
#    ts1 = TensorSum()
#    ts2 = TensorSum()
#    bt.einsum('ijab,kjab->ij', ts1, ts2)
#    # equivalent to:
#    for tensor1 in ts1:
#        for tensor2 in ts2:
#            result += bt.einsum('ijab,kjab->ij', tensor1, tensor2)
