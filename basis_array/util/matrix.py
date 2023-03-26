import numpy as np


__all__ = [
        'Matrix',
        'InverseMatrix',
        'IdentityMatrix',
        'PermutationMatrix',
        'chained_dot',
        ]


class Matrix:

    def __init__(self, values):
        self._values = values
        self._inv = None

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return self.values.shape

    @property
    def values(self):
        return self._values

    @property
    def inv(self):
        if self._inv is None:
            self._inv = InverseMatrix(self)
        return self._inv


class SymmetricMatrix(Matrix):

    @property
    def T(self):
        return self


class InverseMatrix(Matrix):

    def __init__(self, matrix):
        self.matrix = matrix
        self._values = None

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def values(self):
        if self._values is None:
            self._values = np.linalg.inv(getattr(self.matrix, 'values', self.matrix))
        return self._values

    @property
    def inv(self):
        return self.matrix


class IdentityMatrix(SymmetricMatrix):

    def __init__(self, size=None):
        self.size = size

    @property
    def shape(self):
        return (self.size, self.size)

    @property
    def values(self):
        return np.identity(self.size)

    @property
    def inv(self):
        return self


class PermutationMatrix(Matrix):

    def __init__(self, order, size=None):
        self.order = order
        if size is None:
            size = len(order)
        self.size = size

    @property
    def shape(self):
        return (self.size, len(self.order))

    @property
    def values(self):
        return np.identity(self.size)[:, self.order]


def _simplify_matrix_product(a, b, remove_identity=True, remove_inverse=True, remove_permutation=True):
    if a.shape[1] != b.shape[0]:
        raise RuntimeError
    if remove_identity:
        if isinstance(a, IdentityMatrix):
            return [b]
        if isinstance(b, IdentityMatrix):
            return [a]
    if remove_inverse:
        if isinstance(a, InverseMatrix) and a.inv is b:
            return [IdentityMatrix(a.shape[0])]
        if isinstance(b, InverseMatrix) and b.inv is a:
            return [IdentityMatrix(b.shape[0])]
    if remove_permutation:
        if isinstance(a, PermutationMatrix):
            return [Matrix(b.values[np.argsort(a.order)])]
        if isinstance(b, PermutationMatrix):
            return [Matrix(a.values[:, b.order])]
    return [a, b]


def _simplify_n_matrix_products(*matrices, remove_permutation=True):
    if remove_permutation:
        matrices = _simplify_n_matrix_products(*matrices, remove_permutation=False)
    if len(matrices) == 1:
        return matrices
    matrices_out = []
    for i, m in enumerate(matrices[:-1]):
        result = _simplify_matrix_product(m, matrices[i + 1], remove_permutation=remove_permutation)
        matrices_out.append(result[0])
        if len(result) == 2 and (i == len(matrices)-2):
            matrices_out.append(result[1])
        # Restart:
        if len(result) == 1:
            matrices_out.extend(matrices[i+2:])
            return _simplify_n_matrix_products(*matrices_out, remove_permutation=remove_permutation)
    return matrices_out


def chained_dot(*matrices):
    matrices = [m for m in matrices if (m is not None)]
    if len(matrices) == 0:
        return None
    matrices = _simplify_n_matrix_products(*matrices)
    matrices = [getattr(m, 'values', m) for m in matrices]
    if len(matrices) == 1:
        return matrices[0]
    return np.linalg.multi_dot(matrices)
