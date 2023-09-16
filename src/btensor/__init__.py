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

__version__ = '0.0.0'

from loguru import logger
logger.disable('btensor')
#logger.configure(
#    handlers=[
#        dict(sink=sys.stderr, format="[{time}] {message}", colors=True),
#        dict(sink='btensor.log', format="[{time}] {message}")
#    ]
#)

from .basis import IBasis, Basis, nobasis
from .space import Space
from .tensor import Tensor, Cotensor
from .array import Array#, Coarray

from .numpy_functions import *

from .tensorsum import TensorSum
