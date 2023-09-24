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

from __future__ import annotations
from contextlib import contextmanager
from typing import *

import numpy as np

from .matrix import IdentityMatrix

if TYPE_CHECKING:
    from numbers import Number


__all__ = [
        'is_int',
        'is_sequence',
        'array_like',
        'atleast_1d',
        'ndot',
        'expand_axis',
        'replace_attr',
        ]


def is_int(obj):
    return isinstance(obj, (int, np.integer))


def is_sequence(obj: Any) -> bool:
    try:
        len(obj)
        obj[0:0]
    except TypeError:
        return False
    return True

def array_like(obj):
    try:
        obj.shape
        obj.ndim
        obj[()]
        return True
    except (AttributeError, TypeError):
        return False


def atleast_1d(obj):
    return tuple(np.atleast_1d(obj))


def ndot(*args) -> np.ndarray | Number:
    args = [x for x in args if not isinstance(x, IdentityMatrix)]
    args = [a.to_numpy() if hasattr(a, 'to_numpy') else a for a in args]
    return np.linalg.multi_dot(args)


def expand_axis(a, size, indices=None, axis=-1):
    """Expand NumPy array along axis."""
    shape = list(a.shape)
    shape[axis] = size
    if indices is None:
        indices = slice(a.shape[axis])
    if len(np.arange(size)[indices]) != a.shape[axis]:
        raise ValueError
    if axis < 0:
        axis += a.ndim
    mask = axis * (slice(None),) + (indices,) + (a.ndim - axis - 1) * (slice(None),)
    b = np.zeros_like(a, shape=shape)
    b[mask] = a
    return b


@contextmanager
def replace_attr(obj, bind_callable=True, **kwargs):
    """Temporary replace attributes and methods of object."""

    def _setattr(obj, name, attr):
        # For functions: replace and bind as method, otherwise just set
        setattr(obj, name, attr.__get__(obj) if (callable(attr) and bind_callable) else attr)

    orig = {}
    try:
        for name, attr in kwargs.items():
            orig[name] = getattr(obj, name)
            _setattr(obj, name, attr)

        yield obj
    finally:
        # Restore originals
        for name, attr in orig.items():
            _setattr(obj, name, attr)
