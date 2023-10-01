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

import numpy as np

from btensor.tensorsum import TensorSum
from btensor.exceptions import BasisDependentOperationError

if TYPE_CHECKING:
    from btensor import Tensor, IBasis
    EinsumOperandT: TypeAlias = Union[Tensor, TensorSum]


class Einsum:

    def __init__(self,
                 subscripts: str,
                 einsumfunc: Callable = np.einsum,
                 optimize: str | bool = True) -> None:
        # Setup
        self._labels_per_operand, self._result_labels = self._get_labels_per_operand_and_result(subscripts)
        # List of ordered, unique labels
        labels = [x for idx in self._labels_per_operand for x in idx]
        # Remove duplicates while keeping order (sets do not keep order):
        self._unique_labels = list(dict.fromkeys(labels).keys())
        self.einsumfunc = einsumfunc
        self.optimize = optimize

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.get_contraction()})"

    @staticmethod
    def _get_labels_per_operand_and_result(subscripts: str) -> Tuple[List[List[str]], str]:
        # TODO
        if '...' in subscripts:
            raise NotImplementedError("'...' in subscripts not yet supported")
        subscripts = subscripts.replace(' ', '')
        if '->' in subscripts:
            labels, result = subscripts.split('->')
        else:
            labels = subscripts
            # Generate result subscripts automatically: all non-repeated subcripts in alphabetical order
            result = ''.join([s for s in sorted(set(subscripts.replace(',', ''))) if labels.count(s) == 1])
        labels = [list(label) for label in labels.split(',')]
        return labels, result

    @property
    def noperands(self) -> int:
        return len(self._labels_per_operand)

    def get_contraction(self, separator: str = ',', with_result: bool = True) -> str:
        contraction = f"{separator.join([''.join(x) for x in self._labels_per_operand])}"
        if with_result:
            contraction += f"->{self._result_labels}"
        return contraction

    def _contraction_is_basis_independent(self) -> bool:
        joined_labels = self.get_contraction(separator='').replace('->', '')
        for label in self._unique_labels:
            if joined_labels.count(label) != 2:
                return False
        return True

    @staticmethod
    def _get_free_labels(used_labels: List[str]) -> List[str]:
        return sorted(set(string.ascii_letters).difference(set(used_labels)))

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
            result.append(self(*ops, intersect_tol=intersect_tol, **kwargs))
        return TensorSum(result)

    def _label_is_contracted(self, label: str) -> bool:
        return label in self._result_labels

    def _get_unique_contracted_labels(self) -> List[str]:
        return [label for label in self._unique_labels if self._label_is_contracted(label)]

    def _get_positions_of_label(self, label: str) -> List[Tuple[int, int]]:
        positions = []
        for iop, labels in enumerate(self._labels_per_operand):
            for ilab, loop_label in enumerate(labels):
                if label == loop_label:
                    positions.append((iop, ilab))
        return positions

    @overload
    def __call__(self,
                 *operands: Tensor,
                 intersect_tol: Number | None = None,
                 optimize: str | bool | None = None,
                 **kwargs: Any) -> Tensor | Number: ...

    def __call__(self,
                 *operands: EinsumOperandT,
                 intersect_tol: Number | None = None,
                 optimize: str | bool | None = None,
                 **kwargs: Any) -> EinsumOperandT | Number:
        if len(operands) != self.noperands:
            raise ValueError(f"{len(operands)} operands provided, but {self.noperands} specified in subscript string")

        if optimize is None:
            optimize = self.optimize
        kwargs['optimize'] = optimize

        # Support for TensorSums in operands via recursion.
        # This will result in len(TensorSum1) * len(TensorSum2) * ... recursive calls to Einsum
        tensorsums = [(idx, op.to_list()) for idx, op in enumerate(operands) if isinstance(op, TensorSum)]
        if tensorsums:
            return self._resolve_tensorsums(operands, tensorsums, intersect_tol=intersect_tol, **kwargs)

        free_labels = self._get_free_labels(self._unique_labels)
        labels_out = copy.deepcopy(self._labels_per_operand)
        # Loop over all indices
        transformations = []

        basis_out = len(self._result_labels)*[None]
        variance_out = len(self._result_labels)*[None]
        for label in self._unique_labels:
            is_contracted = self._label_is_contracted(label)
            positions = self._get_positions_of_label(label)
            if len(positions) > 2:
                raise BasisDependentOperationError(f"contraction {self.get_contraction()} is basis dependent")

            if is_contracted:
                assert len(positions) == 1
                iop, idim = positions[0]
                idim_result = self._result_labels.index(label)
                op = operands[iop]
                basis_out[idim_result] = op.basis[idim]
                variance_out[idim_result] = op.variance[idim]
            else:
                assert len(positions) == 2
                (iop1, idim1), (iop2, idim2) = positions
                op1 = operands[iop1]
                op2 = operands[iop2]
                bas1 = op1.basis[idim1]
                bas2 = op2.basis[idim2]
                var1 = op1.variance[idim1]
                var2 = op2.variance[idim2]
                # Test if no transformation is needed (same basis and mixed variance or orthonormal)
                if bas1 == bas2 and (bas1.is_orthonormal or (var1 + var2 == 0)):
                    continue
                # Find basis for contraction
                if intersect_tol is not None:
                    bas_contract = bas1.make_intersect_basis(bas2, tol=intersect_tol)
                    # Perform contraction in intersection basis:
                    if bas_contract.size < min(bas1.size, bas2.size):
                        trafo1 = bas1.get_transformation(bas_contract, variance=(-var1, -var2))
                        trafo2 = bas2.get_transformation(bas_contract, variance=(-var2, -var1))
                        new_label1 = free_labels.pop(0)
                        new_label2 = free_labels.pop(0)
                        labels_out[iop1][idim1] = new_label1
                        labels_out[iop2][idim2] = new_label2
                        labels_out.append([new_label1, label])
                        labels_out.append([new_label2, label])
                        transformations.append(trafo1.to_numpy(copy=False))
                        transformations.append(trafo2.to_numpy(copy=False))
                        continue
                # Perform contraction in smaller of bas1, bas2:
                new_label = free_labels.pop(0)
                if bas1.size <= bas2.size:
                    trafo = bas2.get_transformation(bas1, variance=(-var2, -var1))
                    labels_out[iop2][idim2] = new_label
                else:
                    trafo = bas1.get_transformation(bas2, variance=(-var1, -var2))
                    labels_out[iop1][idim1] = new_label
                labels_out.append([new_label, label])
                transformations.append(trafo.to_numpy(copy=False))

        # Return
        subscripts_out = ','.join([''.join(label) for label in labels_out])
        subscripts_out = '->'.join((subscripts_out, self._result_labels))
        operands_out = [op.to_numpy(copy=False) for op in operands]
        operands_out.extend(transformations)
        values = self.einsumfunc(subscripts_out, *operands_out, **kwargs)
        # Contraction result is scalar:
        if not self._result_labels:
            if isinstance(values, np.ndarray):
                assert values.size == 1
                values = values[()]
            return values
        assert (None not in basis_out)
        assert (None not in variance_out)
        cls = type(operands[0])
        return cls(values, basis=tuple(basis_out), variance=tuple(variance_out), copy_data=False)

    def _get_basis_per_label(self, operands: Tuple[EinsumOperandT, ...],
                             intersect_tol: Number | None = None) -> Dict[str, IBasis]:
        basis_per_label = {}
        for unique_label in self._unique_labels:
            basis: IBasis | None = None
            is_output = unique_label in self._result_labels
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

    def __kernel_general(self,
                 *operands: EinsumOperandT,
                 intersect_tol: Number | None = None,
                 **kwargs: Any) -> EinsumOperandT | Number:
        """Older kernel. Slower and does not support non-orthogonal bases, but supports basis dependent operations."""
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
        subscripts_out = '->'.join((subscripts_out, self._result_labels))
        operands_out = [op.to_numpy(copy=False) for op in operands]
        operands_out.extend(transformations)
        values = self.einsumfunc(subscripts_out, *operands_out, **kwargs)
        # Contraction result is scalar:
        if not self._result_labels:
            if isinstance(values, np.ndarray):
                assert values.size == 1
                values = values[()]
            return values
        basis = tuple([basis_per_label[idx] for idx in self._result_labels])
        cls = type(operands[0])
        return cls(values, basis, copy_data=False)


@overload
def einsum(subscripts: str,
           *operands: Tensor,
           intersect_tol: Number | None = None,
           einsumfunc: Callable = np.einsum,
           **kwargs: Any) -> Tensor | Number: ...


def einsum(subscripts: str,
           *operands: EinsumOperandT,
           intersect_tol: Number | None = None,
           einsumfunc: Callable = np.einsum,
           **kwargs: Any) -> EinsumOperandT | Number:
    """Evaluates the Einstein summation on the operands while performing required basis transformations automatically.

    Only basis independent summations are supported. A summation is basis independent, if each label either:

    * Appears once on both the input (before '->') and output side (free label).
    * Appears twice on the input side and not on the output side (contracted label).

    See also the documentation of numpy.einsum.

    Args:
        subscripts: Specifies the subscripts for summation as comma separated list of subscript labels.
        operands: Sequence of tensors for the Einstein summation.
        itersect_tol: If not None, contracted dimensions will first be transformed to the intersect_basis using
            intersect_tol as the truncation tolerance. This may speed up the summation but introduces an error which
            increases with the tolerance. Default: None.
        einsumfunc: Passing this argument allows using a different einsum driver as backend. Default: numpy.einsum.

    Returns:
        Result of the Einstein summation as a tensor or number.

    """
    return Einsum(subscripts, einsumfunc)(*operands, intersect_tol=intersect_tol, **kwargs)
