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
from numbers import Number
from typing import *

from loguru import logger
import numpy as np

from btensor.tensorsum import TensorSum
from btensor.util import BasisDependentOperationError

if TYPE_CHECKING:
    from btensor import Tensor, IBasis
    EinsumOperandT: TypeAlias = Union[Tensor, TensorSum]


class Einsum:

    def __init__(self, subscripts: str, einsumfunc: Callable = np.einsum) -> None:
        self.subscripts = subscripts.replace(' ', '')
        self.einsumfunc = einsumfunc
        # Setup
        self._labels_per_operand, self._result = self._get_labels_per_operand_and_result()
        # List of ordered, unique labels
        labels = [x for idx in self._labels_per_operand for x in idx]
        # Remove duplicates while keeping order (sets do not keep order):
        self._unique_labels = list(dict.fromkeys(labels).keys())
        if not self._contraction_is_basis_independent():
            raise BasisDependentOperationError(f"contraction {self.get_contraction()} is basis dependent")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.get_contraction()})"

    def get_contraction(self) -> str:
        return f"{','.join([''.join(x) for x in self._labels_per_operand])}->{self._result}"

    def _contraction_is_basis_independent(self) -> bool:
        joined_labels = self.get_contraction().replace(',', '').replace('->', '')
        for label in self._unique_labels:
            if joined_labels.count(label) != 2:
                return False
        return True

    def _get_labels_per_operand_and_result(self) -> Tuple[List[List[str]], str]:
        # TODO
        if '...' in self.subscripts:
            raise NotImplementedError("'...' in subscripts not yet supported")
        if '->' in self.subscripts:
            labels, result = self.subscripts.split('->')
        else:
            labels = self.subscripts
            # Generate result subscripts automatically: all non-repeated subcripts in alphabetical order
            result = ''.join([s for s in sorted(set(self.subscripts.replace(',', ''))) if labels.count(s) == 1])
        labels = [list(label) for label in labels.split(',')]
        return labels, result

    @staticmethod
    def _get_free_labels(used_labels: List[str]) -> List[str]:
        return sorted(set(string.ascii_letters).difference(set(used_labels)))

    def _get_basis_per_label(self, operands: Tuple[EinsumOperandT, ...],
                             intersect_tol: Number | None = None) -> Dict[str, IBasis]:
        basis_per_label = {}
        for unique_label in self._unique_labels:
            basis: IBasis | None = None
            is_output = unique_label in self._result
            for operand, operand_labels in zip(operands, self._labels_per_operand):
                # The index might appear multiple times per label -> loop over positions
                positions: List[int] = np.asarray(np.asarray(operand_labels) == unique_label).nonzero()[0]
                for position in positions:
                    current_basis = operand.basis[position]
                    if basis is None:
                        basis = current_basis
                    elif is_output:
                        # Label is in output: find spanning basis
                        basis = basis.get_common_parent(current_basis)
                    else:
                        # Label is contracted
                        if intersect_tol is not None:
                            # Use intersect basis
                            basis = current_basis.make_intersect_basis(basis, tol=intersect_tol)
                        elif current_basis.size < basis.size:
                            # Use smallest basis
                            basis = current_basis
            assert (basis is not None)
            basis_per_label[unique_label] = basis
        return basis_per_label

    def _resolve_tensorsums(self,
                            operands: Tuple[EinsumOperandT, ...],
                            tensorsums: List[Tuple[int, List[Tensor]]],
                            intersect_tol: Number | None = None,
                            **kwargs: Any) -> TensorSum:
        tensorsums_positions, tensor_lists = zip(*tensorsums)
        result = []
        # Loop over all combinations of tensors from the various TensorSums:
        for tensors in itertools.product(*tensor_lists):
            ops = list(operands)
            for tensorsum_idx, pos in enumerate(tensorsums_positions):
                ops[pos] = tensors[tensorsum_idx]
            logger.debug(f"recursive einsum({self.subscripts}) with operands {ops}")
            result.append(self(*ops, intersect_tol=intersect_tol, **kwargs))
        return TensorSum(result)

    @overload
    def __call__(self, *operands: Tensor, **kwargs: Any) -> Tensor | Number: ...

    def __call__(self,
                 *operands: EinsumOperandT,
                 intersect_tol: Number | None = None,
                 **kwargs: Any) -> EinsumOperandT | Number:
        if len(self._labels_per_operand) != len(operands):
            raise ValueError(f"{len(operands)} operands provided, but {len(self._labels_per_operand)} "
                             f"specified in subscript string")

        # Support for TensorSums in operands via recursion.
        # This will result in len(TensorSum1) * len(TensorSum2) * ... recursive calls to Einsum
        tensorsums = [(idx, op.to_list()) for idx, op in enumerate(operands) if isinstance(op, TensorSum)]
        if tensorsums:
            return self._resolve_tensorsums(operands, tensorsums, intersect_tol=intersect_tol, **kwargs)

        free_labels = self._get_free_labels(self._unique_labels)
        labels_out = copy.deepcopy(self._labels_per_operand)
        # Loop over all indices
        transformations = []
        basis_per_label = self._get_basis_per_label(operands, intersect_tol=intersect_tol)
        for unique_label, basis_target in basis_per_label.items():
            # Replace all other bases corresponding to the same index:
            for iop, (op, labels_in_op) in enumerate(zip(operands, self._labels_per_operand)):
                positions_in_op = np.asarray(np.asarray(labels_in_op) == unique_label).nonzero()[0]
                for pos in positions_in_op:
                    basis_current = op.basis[pos]
                    # If the bases are the same, continue, to avoid inserting an unnecessary identity matrix:
                    if basis_current == basis_target:
                        continue
                    # Add transformation from basis_current to basis_target:
                    label_new = free_labels.pop(0)
                    labels_out[iop][pos] = label_new
                    labels_out.append([label_new, unique_label])
                    trafo_variance = (-op.variance[pos], op.variance[pos])
                    trafo = basis_current.get_transformation(basis_target, variance=trafo_variance).to_numpy(copy=False)
                    transformations.append(trafo)

        # Return
        subscripts_out = ','.join([''.join(label) for label in labels_out])
        subscripts_out = '->'.join((subscripts_out, self._result))
        operands_out = [op.to_numpy(copy=False) for op in operands]
        operands_out.extend(transformations)
        values = self.einsumfunc(subscripts_out, *operands_out, **kwargs)
        # Contraction result is scalar:
        if not self._result:
            if isinstance(values, np.ndarray):
                assert values.size == 1
                values = values[()]
            return values
        basis = tuple([basis_per_label[idx] for idx in self._result])
        cls = type(operands[0])
        return cls(values, basis, copy_data=False)


@overload
def einsum(subscripts: str,
           *operands: Tensor,
           einsumfunc: Callable = np.einsum,
           intersect_tol: Number | None = None,
           **kwargs: Any) -> Tensor | Number: ...


def einsum(subscripts: str,
           *operands: EinsumOperandT,
           einsumfunc: Callable = np.einsum,
           intersect_tol: Number | None = None,
           **kwargs: Any) -> EinsumOperandT | Number:
    """Allows contraction of Array objects using Einstein notation.

    The overlap matrices between non-matching dimensions are automatically added.
    """
    return Einsum(subscripts, einsumfunc)(*operands, intersect_tol=intersect_tol, **kwargs)
