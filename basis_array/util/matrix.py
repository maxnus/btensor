import numpy as np
import scipy
import scipy.linalg


DEBUG = True

__all__ = [
        'Matrix',
        'GeneralMatrix',
        'SymmetricMatrix',
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

    def __repr__(self):
        return '%s%r' % (type(self).__name__, self.shape)

    @property
    def ndim(self):
        return 2

    @property
    def inverse(self):
        if self._inverse is None:
            self._inverse = InverseMatrix(self)
        return self._inverse

    # --- Deferred

    @property
    def shape(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError

    @property
    def T(self):
        return self.transpose()

    def to_array(self):
        raise NotImplementedError

    def to_array2(self):
        return self.to_array()


class GeneralMatrix(Matrix):

    def __init__(self, values):
        super().__init__()
        self._transpose = None
        self._values = values

    @property
    def shape(self):
        return self._values.shape

    def to_array(self):
        return self._values

    def transpose(self):
        if self._transpose is None:
            self._transpose = GeneralMatrix(self.to_array().T)
        return self._transpose


class Symmetric:

    def transpose(self):
        return self


class SymmetricMatrix(Symmetric, GeneralMatrix):
    pass


class InverseMatrix(Matrix):

    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix
        self._values = None
        if DEBUG:
            cond = np.linalg.cond(self.matrix.to_array())
            if cond > 1e14:
                raise RuntimeError("Cannot invert matrix %r: condition number= %e" % (self.matrix, cond))

    @property
    def shape(self):
        return self.matrix.shape

    def to_array(self):
        if self._values is None:
            if isinstance(self.matrix, IdentityMatrix):
                self._values = self.matrix.inverse.to_array()
            else:
                self._values = np.linalg.inv(self.matrix.to_array())
        return self._values

    @property
    def inverse(self):
        return self.matrix

    def transpose(self):
        return self.matrix.T.inverse


class IdentityMatrix(Symmetric, Matrix):

    def __init__(self, size):
        super().__init__()
        self.size = size

    @property
    def shape(self):
        return (self.size, self.size)

    def to_array(self):
        return np.identity(self.size)

    @property
    def inverse(self):
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

    def transpose(self):
        return RowPermutationMatrix(self.shape[0], self.permutation)


class RowPermutationMatrix(PermutationMatrix):

    def __init__(self, size, permutation):
        super().__init__(size, permutation)
        self._shape = self._shape[::-1]

    def to_array(self):
        return np.identity(self.shape[1])[self.permutation]

    def transpose(self):
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
        if (isinstance(a, InverseMatrix) and a.inverse is b) or (isinstance(b, InverseMatrix) and b.inverse is a):
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
        if isinstance(a, RowPermutationMatrix) and not isinstance(b, InverseMatrix):
            return [GeneralMatrix(to_array(b)[a.permutation])]
        if isinstance(b, ColumnPermutationMatrix) and not isinstance(a, InverseMatrix):
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
        matrices = [m for m in matrices if m is not None]
        self._matrices = matrices
        for m1, m2 in zip(matrices[:-1], matrices[1:]):
            if m1.shape[1] != m2.shape[0]:
                raise ValueError(f"Invalid matrix product in {self}: {m1.shape} x {m2.shape}")

    def __repr__(self):
        return f'{type(self).__name__}(len ={len(self)}, shape= {self.shape})'

    def __str__(self):
        s = f'{type(self).__name__}('
        s += ' x '.join(f'ndarray{m.shape}' if isinstance(m, np.ndarray) else str(m) for m in self.matrices)
        s += ')'
        return s

    @property
    def matrices(self):
        return self._matrices

    @property
    def shape(self):
        shape = (self.matrices[0].shape[0], self.matrices[-1].shape[-1])
        return shape

    def append(self, matrix):
        self.matrices.append(matrix)

    def extend(self, matrices):
        self.matrices.extend(matrices)

    def insert(self, index, matrix):
        self.matrices.insert(index, matrix)

    def simplify(self):
        if len(self.matrices) < 2:
            return self
        matrices = _simplify_n_matrix_products(self.matrices)
        return MatrixProduct(matrices)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return MatrixProduct(self.matrices[item])
        return self.matrices[item]

    def __len__(self):
        return len(self.matrices)

    def __add__(self, other):
        return MatrixProduct(self.matrices + getattr(other, 'matrices', other))

    def evaluate(self, simplify=True):
        matrices = self.simplify() if simplify else self
        if len(matrices) == 0:
            raise ValueError(f"Cannot evaluate empty {type(self).__name__}")

        # If first matrix A^-1 in A^-1 B... = X is an inverse, solve B... = AX instead
        if isinstance(matrices[0], InverseMatrix) and len(matrices) > 1:
            a = matrices[0].inverse
            b = matrices[1:].evaluate()
            assume_a = 'sym' if isinstance(a, Symmetric) else 'gen'
            return scipy.linalg.solve(a.to_array(), b, assume_a=assume_a)
        # If last matrix Z^-1 in ...Y Z^-1 = X is an inverse, solve (...Y)^T = Z^T X^T instead
        if isinstance(matrices[-1], InverseMatrix) and len(matrices) > 1:
            a = matrices[-1].inverse
            b = matrices[:-1].evaluate()
            assume_a = 'sym' if isinstance(a, Symmetric) else 'gen'
            return scipy.linalg.solve(a.to_array(), b.T, assume_a=assume_a, transposed=True).T

        matrices = [to_array(m) for m in matrices]
        if len(matrices) == 1:
            return matrices[0]
        return np.linalg.multi_dot(matrices)

    def transpose(self):
        return MatrixProduct([m.T for m in reversed(self.matrices)])

    @property
    def T(self):
        return self.transpose()
