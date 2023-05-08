import numpy as np
from btensor import Basis, Tensor


# The standard euclidian 2D basis:
basis1 = Basis(2)
# Rotated basis (x', y') with
# x' = x
# y' = 1\sqrt(2) (x + y)
r = np.asarray([[1, 1/np.sqrt(2)],
                [0, 1/np.sqrt(2)]])
basis2 = Basis(r, parent=basis1)

point1 = Tensor([-1.0, 0.0], basis=basis1)
point2 = Tensor([ 1.0, 1.0], basis=basis2)
point3 = point1 + point2

print(f"Point 3 in basis 1: {point3.to_array(basis=basis1)}")
print(f"Point 3 in basis 2: {point3.to_array(basis=basis2)}")
