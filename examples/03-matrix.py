import numpy as np
from btensor import Basis, Tensor


basis1 = (Basis(2), Basis(3))
basis2 = (Basis([1, 0], parent=basis1[0]),
          Basis([0, 2], parent=basis1[1]))

data1 = np.arange(6).reshape(2, 3)
data2 = np.arange(4).reshape(2, 2)
tensor1 = Tensor(data1, basis=basis1)
tensor2 = Tensor(data2, basis=basis2)

tensor3 = tensor1 + tensor2
print(f"Tensor 3 in basis 1:\n{tensor3.to_array(basis=basis1)}")
print(f"Tensor 3 in basis 2:\n{tensor3.to_array(basis=basis2, project=True)}")
