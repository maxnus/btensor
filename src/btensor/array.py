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

from btensor.util import *
from btensor.basis import Basis
from btensor.tensor import Tensor
from btensor import numpy_functions


class Array(Tensor):

    @property
    def __array_interface__(self):
        return self._data.__array_interface__

    def __getitem__(self, key):
        """Construct and return sub-Array."""

        # getitem of Tensor base class:
        try:
            return super().__getitem__(key)
        except IndexError as e:
            pass

        if isinstance(key, int):
            return type(self)(self._data[key], basis=self.basis[1:])
        if isinstance(key, (list, np.ndarray)):
            value = self._data[key]
            basis = (self.basis[0].make_basis(key),) + self.basis[1:]
            return type(self)(value, basis=basis)
        if isinstance(key, slice) or key is np.newaxis:
            key = (key,)
        if isinstance(key, tuple):
            value = self._data[key]
            if value.ndim == 0:
                return value

            # Add nobasis for each newaxis (None) key
            newaxis_indices = [i for (i, k) in enumerate(key) if (k is np.newaxis)]
            basis = list(self.basis)
            for i in newaxis_indices:
                basis.insert(i, 1)

            # Replace Ellipsis with multiple slice(None)
            if Ellipsis in key:
                idx = key.index(Ellipsis)
                ellipsis_size = len(basis) - len(key) + 1
                key = key[:idx] + ellipsis_size*(slice(None),) + key[idx+1:]

            for i, ki in enumerate(reversed(key), start=1):
                idx = len(key) - i
                if isinstance(ki, (int, np.integer)):
                    del basis[idx]
                elif isinstance(ki, slice):
                    basis[idx] = Basis(argument=ki, parent=basis[idx])
                elif ki is np.newaxis:
                    pass
                else:
                    raise ValueError("key %r of type %r" % (ki, type(ki)))
            basis = tuple(basis)
            return type(self)(value, basis=basis)
        raise NotImplementedError("Key= %r of type %r" % (key, type(key)))

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value._data
        with replace_attr(self._data.flags, writeable=True):
            self._data[key] = value
        # Not required, since np.newaxis has no effect in assignment?
        #if not isinstance(key, tuple) or np.newaxis not in key:
        #    return
        #basis_old = list(self.basis)
        #basis_new = tuple(nobasis if elem is np.newaxis else basis_old.pop(0) for elem in key)
        #self.basis = basis_new

    def to_tensor(self):
        return Tensor(self._data, basis=self.basis)

    def sum(self, axis=None):
        return numpy_functions.sum(self, axis=axis)


#class Coarray(Array, Cotensor):
#    pass
