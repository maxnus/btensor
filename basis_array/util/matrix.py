import numpy as np


__all__ = [
        'Matrix',
        'IdentityMatrix',
        'InverseMatrix',
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


def _remove_identity(matrices):
    matrices_out = []
    for i, m in enumerate(matrices):
        if isinstance(m, IdentityMatrix):
            if m.size is None:
                continue
            # Check dimension of matrices:
            m_prev = matrices[max(i-1, 0)]
            m_next = matrices[min(i+1, len(matrices)-1)]
            if m.size != m_prev.shape[-1]:
                raise ValueError("Cannot multiply matrices with shapes %r and %r" % (m_prev.shape, m.shape))
            if m.size != m_next.shape[max(-2, -m_next.ndim)]:
                raise ValueError("Cannot multiply matrices with shapes %r and %r" % (m.shape, m_next.shape))
        else:
            matrices_out.append(m)
    return matrices_out


def _remove_inverses(matrices):
    matrices = matrices.copy()
    for i, m in enumerate(matrices):
        if not hasattr(m, 'inv'):
            continue
        if i > 0 and m.inv is matrices[i-1]:
            matrices.pop(i)
            matrices.pop(i-1)
            break
        if i < len(matrices)-1 and m.inv is matrices[i+1]:
            matrices.pop(i+1)
            matrices.pop(i)
            break
    else:
        return matrices
    return _remove_inverses(matrices)


def _replace_matrices(matrices):
    return [(m.values if isinstance(m, Matrix) else m) for m in matrices]


def chained_dot(*matrices):
    matrices = [m for m in matrices if (m is not None)]
    if len(matrices) == 0:
        return None
    shape_out = matrices[0].shape[0], matrices[-1].shape[-1]
    matrices = _remove_identity(matrices)
    matrices = _remove_inverses(matrices)
    matrices = _replace_matrices(matrices)
    #print("Chained dot with %d matrices" % len(matrices))
    if len(matrices) == 0:
        assert shape_out[0] == shape_out[1]
        return np.identity(shape_out[0])
    if len(matrices) == 1:
        return matrices[0]
    return np.linalg.multi_dot(matrices)
