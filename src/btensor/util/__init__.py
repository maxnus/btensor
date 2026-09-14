#     Copyright 2023-2026 Max Nusspickel
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


from .matrix import (
    ColumnPermutationMatrix,
    GeneralMatrix,
    IdentityMatrix,
    InverseMatrix,
    Matrix,
    MatrixProductList,
    PermutationMatrix,
    RowPermutationMatrix,
    SymmetricMatrix,
    to_numpy,
)
from .util import (
    array_like,
    atleast_1d,
    check_input,
    expand_axis,
    is_int,
    is_sequence,
    ndot,
    replace_attr,
    text_enumeration,
)

__all__ = [
    "ColumnPermutationMatrix",
    "GeneralMatrix",
    "IdentityMatrix",
    "InverseMatrix",
    "Matrix",
    "MatrixProductList",
    "PermutationMatrix",
    "RowPermutationMatrix",
    "SymmetricMatrix",
    "array_like",
    "atleast_1d",
    "check_input",
    "expand_axis",
    "is_int",
    "is_sequence",
    "ndot",
    "replace_attr",
    "text_enumeration",
    "to_numpy",
]
