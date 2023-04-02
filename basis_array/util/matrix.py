import numpy as np


DEBUG = True

__all__ = [
        'Matrix',
        'InverseMatrix',
        'IdentityMatrix',
        'PermutationMatrix',
        'chained_dot',
        ]


def to_array(object):
    if hasattr(object, 'to_array'):
        return object.to_array()
    return object

class Matrix:

    def __init__(self, values):
        self._values = values
        self._inv = None

    def __repr__(self):
        return '%s(shape= %r)' % (type(self).__name__, self.shape)

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return self._values.shape

    def to_array(self):
        return self._values

    @property
    def inv(self):
        if self._inv is None:
            self._inv = InverseMatrix(self)
        return self._inv

    @property
    def T(self):
        return Matrix(self.to_array().T)


class SymmetricMatrix(Matrix):

    @property
    def T(self):
        return self


class InverseMatrix(Matrix):

    def __init__(self, matrix):
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
            self._values = np.linalg.inv(to_array(self.matrix))
        return self._values

    @property
    def inv(self):
        return self.matrix


class IdentityMatrix(SymmetricMatrix):

    def __init__(self, size):
        self.size = size

    def __repr__(self):
        return '%s(size= %r)' % (type(self).__name__, self.size)

    @property
    def shape(self):
        return (self.size, self.size)

    def to_array(self):
        return np.identity(self.size)

    @property
    def inv(self):
        return self


class PermutationMatrix(Matrix):

    def __init__(self, size, permutation, axis=1):
        if isinstance(permutation, slice):
            shape = (size, len(np.arange(size)[permutation]))
        else:
            shape = (size, len(permutation))
        if axis == 0:
            shape = shape[::-1]
        self._shape = shape
        self._permutation = permutation
        self._axis = axis

    def __repr__(self):
        return '%s(shape= %r, axis= %d)' % (type(self).__name__, self.shape, self.axis)

    @property
    def shape(self):
        return self._shape

    @property
    def axis(self):
        return self._axis

    @property
    def permutation(self):
        return self._permutation

    def to_array(self):
        if self.axis == 1:
            return np.identity(self.shape[0])[:, self.permutation]
        if self.axis == 0:
            return np.identity(self.shape[1])[self.permutation]

    @property
    def T(self):
        if isinstance(self.permutation, slice):
            axis = 0 if (self.axis == 1) else 1
            permutation = np.arange(self.shape[axis])[self.permutation]
        else:
            permutation = self.permutation
        if self.axis == 1:
            p = PermutationMatrix(self.shape[0], permutation, axis=0)
        elif self.axis == 0:
            p = PermutationMatrix(self.shape[1], permutation, axis=1)
        return p


def _simplify_matrix_product(a, b, remove_identity=True, remove_inverse=True, remove_permutation=True):
    if a.shape[1] != b.shape[0]:
        raise RuntimeError("Cannot perform matrix product: %r x %r" % (a.shape, b.shape))
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
        if isinstance(a, PermutationMatrix):
            if a.axis == 0:
                return [Matrix(b.to_array()[a.permutation])]
            elif a.axis == 1:
                return [Matrix(b.to_array()[np.argsort(a.permutation)])]
        if isinstance(b, PermutationMatrix):
            if b.axis == 1:
                return [Matrix(a.to_array()[:, b.permutation])]
            elif b.axis == 0:
                # NOT TESTED
                #return [Matrix(a.to_array()[:. np.argsort(a.permutation)])]
                raise NotImplementedError
    return [a, b]


def _simplify_n_matrix_products(*matrices, remove_permutation=True):
    if remove_permutation:
        matrices = _simplify_n_matrix_products(*matrices, remove_permutation=False)
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
            return _simplify_n_matrix_products(*matrices_out, remove_permutation=remove_permutation)
    return matrices_out


def chained_dot(*matrices):
    matrices = [m for m in matrices if (m is not None)]
    #for i, m in enumerate(matrices):
    #    print('Matrix %d of type %s: %r' % (i, type(m), m))
    #    print(to_array(m))
    if len(matrices) == 0:
        return None
    matrices = _simplify_n_matrix_products(*matrices)
    matrices = [to_array(m) for m in matrices]
    if len(matrices) == 1:
        return matrices[0]
    return np.linalg.multi_dot(matrices)
