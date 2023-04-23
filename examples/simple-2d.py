import numpy as np
from basis_array import Basis, Tensor


# The standard euclidian 2D basis:
basis1 = Basis(2)
# Rotated basis (x', y') with
# x' = x
# y' = 1\sqrt(2) (x + y)
r = np.asarray([[1, 1/np.sqrt(2)],
                [0, 1/np.sqrt(2)]])
basis2 = Basis(r, parent=basis1)

point1 = Tensor([1.0, 0.0], basis=basis1)
point2 = Tensor([0.0, 1.0], basis=basis2)
point3 = point1 + point2

print("Point 3 in basis 1: %r" % (point3 | basis1).value)
print("Point 3 in basis 2: %r" % (point3 | basis2).value)
