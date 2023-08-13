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
from typing import *

import numpy as np

if TYPE_CHECKING:
    from btensor import Tensor


class TensorSum:
    """Class for delayed tensor addition."""

    def __init__(self, tensors: list[Tensor], allow_combine: bool = False) -> None:
        self._tensors = []
        self._allow_combine = allow_combine
        for tensor in tensors:
            self.add_tensor(tensor)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size= {len(self)})"

    def info(self) -> str:
        info = f"{repr(self)} ["
        if len(self.tensors):
            info += "\n"
        for tensor in self.tensors:
            info += f"    {repr(tensor)},\n"
        info += "]\n"
        return info

    @property
    def tensors(self) -> List[Tensor]:
        return self._tensors

    def add_tensor(self, tensor: Tensor, allow_combine: bool | None = None) -> None:
        if allow_combine is None:
            allow_combine = self._allow_combine
        if len(self):
            if self.tensors[0].basis.get_root_basistuple() != tensor.basis.get_root_basistuple():
                raise ValueError
        if allow_combine:
            for idx, tensor_super in enumerate(self.tensors):
                if tensor_super.basis.is_spanning(tensor.basis):
                    self.tensors[idx] = (tensor_super + tensor)
                    return
        self.tensors.append(tensor)

    def __len__(self) -> int:
        return len(self.tensors)

    @overload
    def __getitem__(self, item: int) -> Tensor: ...

    @overload
    def __getitem__(self, item: slice) -> TensorSum: ...

    def __getitem__(self, item: int | slice) -> Tensor | TensorSum:
        return self.tensors[item]

    def evaluate(self) -> Tensor:
        return sum(self.tensors)

    def to_numpy(self) -> np.ndarray:
        return self.evaluate().to_numpy()

    def dot(self, other: Tensor | TensorSum) -> TensorSum:
        out = TensorSum([])
        for tensor in self.tensors:
            if isinstance(other, TensorSum):
                for tensor2 in other.tensors:
                    out.add_tensor(tensor.dot(tensor2))
            else:
                out.add_tensor(tensor.dot(other))
        return out
