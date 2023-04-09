import numpy as np


DEBUG = True

__all__ = [
        'Matrix',
        'GeneralMatrix',
        'InverseMatrix',
        'IdentityMatrix',
        'RowPermutationMatrix',
        'ColumnPermutationMatrix',
        'MatrixProduct',
        'to_array',
        ]


def to_array(object):
    if hasattr(object, 'to_array'):
        return object.to_array()
    return object


class Matrix:

    def __init__(self):
        self._inverse = None
        self._transpose = None

    def __repr__(self):
        return '%s(shape= %r)' % (type(self).__name__, self.shape)

    @property
    def ndim(self):
        return 2

    @property
    def inv(self):
        if self._inverse is None:
            self._inverse = InverseMatrix(self)
        return self._inverse

    # --- Deferred

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def T(self):
        raise NotImplementedError

    def to_array(self):
        raise NotImplementedError


class GeneralMatrix(Matrix):

    def __init__(self, values):
        super().__init__()
        self._values = values

    @property
    def shape(self):
        return self._values.shape

    def to_array(self):
        return self._values

    @property
    def T(self):
        if self._transpose is None:
            self._transpose = GeneralMatrix(self.to_array().T)
        return self._transpose


class SymmetricMatrix(Matrix):

    @property
    def T(self):
        return self


class InverseMatrix(Matrix):

    def __init__(self, matrix):
        super().__init__()
        if DEBUG:
            cond = np.linalg.cond(to_array(matrix))
            if cond > 1e14:
                raise RuntimeError("Cannot invert matrix %r: condition number= %e" % (matrix, cond))
        self.matrix = matrix
        self._values = None

    @property
    def shape(self):
        return self.matrix.shape

    def to_array(self):
        if self._values is None:
            if isinstance(self.matrix, IdentityMatrix):
                self._values = self.matrix.inv.to_array()
            else:
                self._values = np.linalg.inv(to_array(self.matrix))
        return self._values

    @property
    def inv(self):
        return self.matrix

    @property
    def T(self):
        return InverseMatrix(self.matrix.T)


class IdentityMatrix(SymmetricMatrix):

    def __init__(self, size):
        super().__init__()
        self.size = size

    @property
    def shape(self):
        return (self.size, self.size)

    def to_array(self):
        return np.identity(self.size)

    @property
    def inv(self):
        return self


class PermutationMatrix(Matrix):

    def __init__(self, size, permutation):
        super().__init__()
        if isinstance(permutation, slice):
            nperm = len(np.arange(size)[permutation])
        else:
            nperm = len(permutation)
        self._shape = (size, nperm)
        self._permutation = permutation

    @property
    def shape(self):
        return self._shape

    @property
    def permutation(self):
        return self._permutation


class ColumnPermutationMatrix(PermutationMatrix):

    def to_array(self):
        return np.identity(self.shape[0])[:, self.permutation]

    @property
    def T(self):
        return RowPermutationMatrix(self.shape[0], self.permutation)


class RowPermutationMatrix(PermutationMatrix):

    def __init__(self, size, permutation):
        super().__init__(size, permutation)
        self._shape = self._shape[::-1]

    def to_array(self):
        return np.identity(self.shape[1])[self.permutation]

    @property
    def T(self):
        return ColumnPermutationMatrix(self.shape[1], self.permutation)


def _simplify_matrix_product(a, b, remove_identity=True, remove_inverse=True, remove_permutation=True):
    if a.shape[1] != b.shape[0]:
        raise RuntimeError("Cannot take matrix product between matrices with shape: %r x %r" % (a.shape, b.shape))
    if remove_identity:
        if isinstance(a, IdentityMatrix):
            return [b]
        if isinstance(b, IdentityMatrix):
            return [a]
    if remove_inverse:
        if (isinstance(a, InverseMatrix) and a.inv is b) or (isinstance(b, InverseMatrix) and b.inv is a):
            assert(a.shape[0] == b.shape[1])
            return [IdentityMatrix(a.shape[0])]
    if remove_permutation:
        # Combine permutation matrices
        if isinstance(a, ColumnPermutationMatrix) and isinstance(b, ColumnPermutationMatrix):
            permutation = a.permutation[b.permutation]
            return [ColumnPermutationMatrix(permutation=permutation, size=a.shape[0])]
        if isinstance(a, RowPermutationMatrix) and isinstance(b, RowPermutationMatrix):
            permutation = b.permutation[a.permutation]
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
        if isinstance(a, RowPermutationMatrix):
            return [GeneralMatrix(to_array(b)[a.permutation])]
        if isinstance(b, ColumnPermutationMatrix):
            return [GeneralMatrix(to_array(a)[:, b.permutation])]

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


class MatrixProduct:

    def __init__(self, matrices):
        self._matrices = [m for m in matrices if m is not None]

    @property
    def matrices(self):
        return self._matrices

    @property
    def shape(self):
        shape = (self.matrices[0].shape[0], self.matrices[-1].shape[-1])
        return shape

    def append(self, matrix):
        self.matrices.append(matrix)

    def insert(self, index, matrix):
        self.matrices.insert(index, matrix)

    def simplify(self):
        if len(self.matrices) < 2:
            return self
        matrices = _simplify_n_matrix_products(self.matrices)
        return MatrixProduct(matrices)

    def __getitem__(self, item):
        return self.matrices[item]

    def __len__(self):
        return len(self.matrices)

    def __add__(self, other):
        return MatrixProduct(self.matrices + getattr(other, 'matrices', other))

    def evaluate(self, simplify=True):
        matrices = self.simplify() if simplify else self.matrices
        matrices = [to_array(m) for m in matrices]
        if len(matrices) == 0:
            raise ValueError("Cannot evaluate empty %s" % type(self).__name__)
        if len(matrices) == 1:
            return matrices[0]
        return np.linalg.multi_dot(matrices)

    def transpose(self):
        return

    @property
    def T(self):
        return self.transpose()
