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


class BTensorError(Exception):
    pass


class BasisError(BTensorError):
    pass


class VarianceError(BTensorError):
    pass


class BasisDependentOperationError(BasisError):

    def __init__(self, msg: str = "operation may be basis dependent; set allow_bdo to True to allow", *args) -> None:
        super().__init__(msg, *args)
