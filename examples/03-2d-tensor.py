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


basis1 = (Basis(2), Basis(3))
basis2 = (basis1[0].make_subbasis([1, 0]),
          basis1[1].make_subbasis([0, 2]))

data1 = np.arange(6).reshape(2, 3)
data2 = np.arange(4).reshape(2, 2)
tensor1 = Tensor(data1, basis=basis1)
tensor2 = Tensor(data2, basis=basis2)

tensor3 = tensor1 + tensor2
print(f"tensor3 in basis1:\n{tensor3[basis1].to_numpy()}")
print(f"tensor3 in basis2:\n{tensor3[basis2].to_numpy()}")
