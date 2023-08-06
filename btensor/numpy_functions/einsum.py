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
import copy
import string
import itertools
from typing import *

from loguru import logger
import numpy as np

from btensor.tensorsum import TensorSum

if TYPE_CHECKING:
    from btensor import Tensor, BasisInterface


class Einsum:

    def __init__(self, subscripts: str, einsumfunc: Callable = np.einsum) -> None:
        self.subscripts = subscripts.replace(' ', '')
        self.einsumfunc = einsumfunc
        # Setup
        self.labels, self.result = self._get_labels_and_result()
        # Sorted set of all indices
        indices = [x for idx in self.labels for x in idx]
        # Remove duplicates while keeping order (sets do not keep order):
        self.indices = list(dict.fromkeys(indices).keys())

    def _get_labels_and_result(self) -> Tuple[List[List[str]], str]:
        if '...' in self.subscripts:
            raise NotImplementedError("... not currently supported")
        if '->' in self.subscripts:
            labels, result = self.subscripts.split('->')
        else:
            labels = self.subscripts
            # Generate result subscripts automatically: all non-repeated subcripts in alphabetical order
            result = ''.join([s for s in sorted(set(self.subscripts.replace(',', ''))) if labels.count(s) == 1])
        labels = [list(label) for label in labels.split(',')]
        return labels, result

    def _get_free_indices(self) -> List[str]:
        return sorted(set(string.ascii_letters).difference(set(self.indices)))

    @staticmethod
    def _get_basis_target(index: str, operands: Tuple[Tensor, ...], labels: List[List[str]]) -> BasisInterface:
        # Find smallest basis for given idx:
        basis_target = None
        for i, label in enumerate(labels):
            # The index might appear multiple times per label -> loop over positions
            positions = np.asarray(np.asarray(label) == index).nonzero()[0]
            for pos in positions:
                basis_current = operands[i].basis[pos]
                if basis_target is None or (basis_current.size < basis_target.size):
                    basis_target = basis_current
        assert (basis_target is not None)
        return basis_target

    def __call__(self, *operands: Tensor | TensorSum, **kwargs) -> Tensor | TensorSum:
        if len(self.labels) != len(operands):
            raise ValueError("invalid number of operands")

        # Support for TensorSum via recursion
        tensorsums = [(idx, list(op)) for idx, op in enumerate(operands) if isinstance(op, TensorSum)]
        if tensorsums:
            tensorsums_positions, tensorsums = zip(*tensorsums)
            result = []
            # Loop over all combinations of tensors from the various TensorSums:
            for tensors in itertools.product(*tensorsums):
                ops = list(operands)
                for tensorsum_idx, pos in enumerate(tensorsums_positions):
                    ops[pos] = tensors[tensorsum_idx]
                logger.debug(f"recursive einsum({self.subscripts}) with operands {ops}")
                result.append(Einsum(self.subscripts, self.einsumfunc)(*ops, **kwargs))
            return TensorSum(result)

        free_indices = self._get_free_indices()
        labels_out = copy.deepcopy(self.labels)
        # Loop over all indices
        overlaps = []
        basis_per_index = {}
        for index in self.indices:
            # Find smallest basis for given idx:
            basis_target = self._get_basis_target(index, operands, self.labels)
            basis_per_index[index] = basis_target

            # Replace all other bases corresponding to the same index:
            for i, label in enumerate(self.labels):
                positions = np.asarray(np.asarray(label) == index).nonzero()[0]
                for pos in positions:
                    basis_current = operands[i].basis[pos]
                    # If the bases are the same, continue, to avoid inserting an unnecessary identity matrix:
                    if basis_current == basis_target:
                        continue
                    # Add transformation from basis_current to basis_target:
                    index_new = free_indices.pop(0)
                    labels_out[i][pos] = index_new
                    labels_out.append([index_new, index])
                    overlaps.append(basis_current.get_overlap(basis_target).to_numpy(copy=False))

        # Return
        subscripts_out = ','.join([''.join(label) for label in labels_out])
        subscripts_out = '->'.join((subscripts_out, self.result))
        operands_out = [op.to_numpy(copy=False) for op in operands]
        operands_out.extend(overlaps)
        values = self.einsumfunc(subscripts_out, *operands_out, **kwargs)
        basis_out = tuple([basis_per_index[idx] for idx in self.result])
        cls = type(operands[0])
        return cls(values, basis_out)


def einsum(subscripts: str, *operands: Tensor | TensorSum, einsumfunc: Callable = np.einsum, **kwargs) -> Tensor:
    """Allows contraction of Array objects using Einstein notation.

    The overlap matrices between non-matching dimensions are automatically added.
    """
    return Einsum(subscripts, einsumfunc)(*operands, **kwargs)
