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
from btensor import Basis


rootbasis = Basis(3)

# --- A derived basis can be constructed in terms of
# 1) A general transformation matrix
tm = np.asarray([[1, 0],
                 [0, 0],
                 [0, 1]])
basis1 = Basis(tm, parent=rootbasis)
# 2) an indexing array
basis2 = Basis([0, 2], parent=rootbasis)
# 3) a slice object
basis3 = Basis(slice(0, 3, 2), parent=rootbasis)
# 4) a masking array
basis4 = Basis([True, False, True], parent=rootbasis)

# --- Note that all the definitions above are equivalent:
assert np.allclose(basis1.get_transformation_to(basis2).to_numpy(), np.identity(2))
assert np.allclose(basis1.get_transformation_to(basis3).to_numpy(), np.identity(2))
assert np.allclose(basis1.get_transformation_to(basis4).to_numpy(), np.identity(2))
