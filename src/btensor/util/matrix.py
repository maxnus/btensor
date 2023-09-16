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
from collections import UserList
from collections.abc import MutableSequence

import numpy as np
import scipy
import scipy.linalg


__all__ = [
    'Matrix',
    'GeneralMatrix',
    'SymmetricMatrix',
    'InverseMatrix',
    'IdentityMatrix',
    'PermutationMatrix',
    'RowPermutationMatrix',
    'ColumnPermutationMatrix',
    'MatrixProductList',
    'to_numpy',
]


def to_numpy(obj):
    if hasattr(obj, 'to_numpy'):
        return obj.to_numpy()
    return obj


class Matrix:

    def __init__(self) -> None:
        self._inverse = None

    def __repr__(self) -> str:
        return '%s%r' % (type(self).__name__, self.shape)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def inverse(self) -> InverseMatrix:
        if self._inverse is None:
            self._inverse = InverseMatrix(self)
        return self._inverse

    # --- Deferred

    @property
    def shape(self) -> tuple[int, int]:
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError

    @property
    def T(self):
        return self.transpose()

    def to_numpy(self):
        raise NotImplementedError


class GeneralMatrix(Matrix):

    def __init__(self, values):
        super().__init__()
        self._transpose = None
        self._values = values

    @property
    def shape(self) -> tuple[int, int]:
        return self._values.shape

    def to_numpy(self):
        return self._values

    def transpose(self):
        if self._transpose is None:
            self._transpose = GeneralMatrix(self.to_numpy().T)
        return self._transpose


class Symmetric:

    def transpose(self):
        return self


class SymmetricMatrix(Symmetric, GeneralMatrix):
    pass


class InverseMatrix(Matrix):

    def __init__(self, matrix) -> None:
        super().__init__()
        self.matrix = matrix
        self._values = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def to_numpy(self):
        if self._values is None:
            if isinstance(self.matrix, IdentityMatrix):
                self._values = self.matrix.inverse.to_numpy()
            else:
                self._values = np.linalg.inv(self.matrix.to_numpy())
        return self._values

    @property
    def inverse(self):
        return self.matrix

    def transpose(self):
        return self.matrix.T.inverse


class IdentityMatrix(Symmetric, Matrix):

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    def to_numpy(self):
        return np.identity(self.size)

    @property
    def inverse(self):
        return self


class PermutationMatrix(Matrix):

    def __init__(self, size: int, permutation: slice | Sequence[int]) -> None:
        super().__init__()
        if isinstance(permutation, slice):
            nperm = len(np.arange(size)[permutation])
        else:
            permutation = np.asarray(permutation, dtype=int)
            nperm = len(permutation)
        self._shape = (size, nperm)
        self._permutation = permutation

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def permutation(self):
        return self._permutation


class ColumnPermutationMatrix(PermutationMatrix):

    def to_numpy(self):
        return np.identity(self.shape[0])[:, self.permutation]

    def transpose(self):
        return RowPermutationMatrix(self.shape[0], self.permutation)

    @property
    def indices(self):
        if isinstance(self.permutation, slice):
            return np.arange(self.shape[0])[self.permutation]
        return self.permutation


class RowPermutationMatrix(PermutationMatrix):

    def __init__(self, size, permutation):
        super().__init__(size, permutation)
        self._shape = self._shape[::-1]

    def to_numpy(self):
        return np.identity(self.shape[1])[self.permutation]

    def transpose(self):
        return ColumnPermutationMatrix(self.shape[1], self.permutation)

    @property
    def indices(self):
        if isinstance(self.permutation, slice):
            return np.arange(self.shape[1])[self.permutation]
        return self.permutation


def _simplify_matrix_product(a, b, remove_identity=True, remove_inverse=True, remove_permutation=True):
    if a.shape[1] != b.shape[0]:
        raise RuntimeError("Cannot take matrix product between matrices with shape: %r x %r" % (a.shape, b.shape))
    if remove_identity:
        if isinstance(a, IdentityMatrix):
            return [b]
        if isinstance(b, IdentityMatrix):
            return [a]
    if remove_inverse:
        if (isinstance(a, InverseMatrix) and a.inverse is b) or (isinstance(b, InverseMatrix) and b.inverse is a):
            assert(a.shape[0] == b.shape[1])
            return [IdentityMatrix(a.shape[0])]
    if remove_permutation:
        # Combine permutation matrices
        if isinstance(a, ColumnPermutationMatrix) and isinstance(b, ColumnPermutationMatrix):
            permutation = a.indices[b.permutation]
            return [ColumnPermutationMatrix(permutation=permutation, size=a.shape[0])]
        if isinstance(a, RowPermutationMatrix) and isinstance(b, RowPermutationMatrix):
            permutation = b.indices[a.permutation]
            return [RowPermutationMatrix(permutation=permutation, size=b.shape[1])]
        # TODO
        # Combine Row with Column permutation matrices
        #if isinstance(a, RowPermutationMatrix) and isinstance(b, ColumnPermutationMatrix):
        #    if a.shape[0] >= b.shape1[1]:
        #        #permutation = np.argsort(a.permutation)[b.permutation]
        #        print(a.permutation)
        #        print(b.permutation)
        #        permutation = np.argsort(a.permutation[np.argsort(b.permutation)])
        #        print(permutation)
        #        print(a.shape, b.shape)
        #        return [ColumnPermutationMatrix(permutation=permutation, size=a.shape[0])]
        #if isinstance(a, ColumnPermutationMatrix) and isinstance(b, RowPermutationMatrix):
        #    permutation = a.permutation[np.argsort(b.permutation)]
        #    return [ColumnPermutationMatrix(permutation=permutation, size=a.shape[0])]
        #    #permutation = b.permutation[np.argsort(a.permutation)]
        #    #return [RowPermutationMatrix(permutation=permutation, size=a.shape[1])]

        # NEW
        if isinstance(a, RowPermutationMatrix) and not isinstance(b, InverseMatrix):
            return [GeneralMatrix(to_numpy(b)[a.permutation])]
        if isinstance(b, ColumnPermutationMatrix) and not isinstance(a, InverseMatrix):
            return [GeneralMatrix(to_numpy(a)[:, b.permutation])]

    return [a, b]


def _simplify_n_matrix_products(matrices, remove_permutation=True):
    if remove_permutation:
        matrices = _simplify_n_matrix_products(matrices, remove_permutation=False)
    if len(matrices) == 1:
        return matrices
    matrices_out = []
    for i, m in enumerate(matrices[:-1]):
        result = _simplify_matrix_product(m, matrices[i + 1], remove_permutation=remove_permutation)
        if result:
            matrices_out.append(result[0])
        if len(result) == 2 and (i == len(matrices)-2):
            matrices_out.append(result[1])
        # Restart:
        if len(result) < 2:
            matrices_out.extend(matrices[i+2:])
            return _simplify_n_matrix_products(matrices_out, remove_permutation=remove_permutation)
    return matrices_out


#class MatrixProductList(UserList):
class MatrixProductList(list):

    def __init__(self, matrices: MutableSequence[Matrix]) -> None:
        self.check_if_matrix(*matrices)
        self.check_valid_shapes(matrices)
        super().__init__(matrices)

    def check_if_matrix(self, *matrices: Matrix) -> None:
        for matrix in matrices:
            if not isinstance(matrix, Matrix):
                raise TypeError(f"only type {Matrix.__name__} allowed in {type(self).__name__} (not {matrix})")

    def check_valid_shapes(self, matrices: MutableSequence[Matrix]) -> None:
        for m1, m2 in zip(matrices[:-1], matrices[1:]):
            if m1.shape[1] != m2.shape[0]:
                raise ValueError(f"Invalid matrix product in {self}: {m1.shape} x {m2.shape}")

    def __repr__(self) -> str:
        return type(self).__name__ + super().__repr__()

    @property
    def shape(self) -> tuple[int, int]:
        if len(self) == 0:
            raise RuntimeError(f"{type(self).__name__} is empty")
        shape = (self[0].shape[0], self[-1].shape[-1])
        return shape

    def append(self, matrix: Matrix) -> None:
        self.check_if_matrix(matrix)
        super().append(matrix)

    def extend(self, matrices: MutableSequence[Matrix]) -> None:
        if not isinstance(matrices, MatrixProductList):
            self.check_if_matrix(*matrices)
        super().extend(matrices)

    def insert(self, index: int, matrix: Matrix) -> None:
        self.check_if_matrix(matrix)
        super().insert(index, matrix)

    def simplify(self) -> MatrixProductList:
        if len(self) < 2:
            return self
        matrices = _simplify_n_matrix_products(self)
        return type(self)(matrices)

    @overload
    def __getitem__(self, key: int) -> Matrix: ...

    @overload
    def __getitem__(self, key: slice) -> MatrixProductList: ...

    def __getitem__(self, key: int | slice) -> Matrix | MatrixProductList:
        result = super().__getitem__(key)
        if isinstance(result, list):
            return MatrixProductList(result)
        return result

    def __add__(self, other: list[Matrix] | MatrixProductList) -> MatrixProductList:
        if not isinstance(other, MatrixProductList):
            self.check_if_matrix(*other)
        return type(self)(super().__add__(other))

    def evaluate(self, simplify: bool = True) -> np.ndarray:
        matrices = self.simplify() if simplify else self
        if len(matrices) == 0:
            raise ValueError(f"Cannot evaluate empty {type(self).__name__}")
        if len(matrices) > 1:
            # If first matrix A^-1 in A^-1 B... = X is an inverse, solve B... = AX instead
            if isinstance(matrices[0], InverseMatrix):
                a = matrices[0].inverse
                b = matrices[1:].evaluate()
                assume_a = 'sym' if isinstance(a, Symmetric) else 'gen'
                return scipy.linalg.solve(a.to_numpy(), b, assume_a=assume_a)
            # If last matrix Z^-1 in ...Y Z^-1 = X is an inverse, solve (...Y)^T = Z^T X^T instead
            if isinstance(matrices[-1], InverseMatrix):
                a = matrices[-1].inverse
                b = matrices[:-1].evaluate()
                assume_a = 'sym' if isinstance(a, Symmetric) else 'gen'
                return scipy.linalg.solve(a.to_numpy(), b.T, assume_a=assume_a, transposed=True).T

        matrices = [to_numpy(m) for m in matrices]
        if len(matrices) == 1:
            return matrices[0]
        return np.linalg.multi_dot(matrices)

    def transpose(self) -> MatrixProductList:
        return type(self)([m.T for m in reversed(self)])

    @property
    def T(self) -> MatrixProductList:
        return self.transpose()
