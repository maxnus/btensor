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
point2 = Tensor([1.0, 1.0], basis=basis2)
point3 = point1 + point2

print(f"Point 3 in basis 1: {point3.to_numpy(basis=basis1)}")
print(f"Point 3 in basis 2: {point3.to_numpy(basis=basis2)}")
