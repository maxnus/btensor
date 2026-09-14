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

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from .matrix import IdentityMatrix

if TYPE_CHECKING:
    from numbers import Number


__all__ = [
    "array_like",
    "atleast_1d",
    "check_input",
    "expand_axis",
    "is_int",
    "is_sequence",
    "ndot",
    "replace_attr",
    "text_enumeration",
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
    # The attribute and item accesses below are probes: they raise for objects
    # that do not look like arrays, which is exactly what is being tested.
    try:
        obj.shape  # noqa: B018
        obj.ndim  # noqa: B018
        obj[()]
        return True
    except (AttributeError, TypeError):
        return False


def atleast_1d(obj):
    return tuple(np.atleast_1d(obj))


def ndot(*args) -> np.ndarray | Number:
    args = [x for x in args if not isinstance(x, IdentityMatrix)]
    args = [a.to_numpy() if hasattr(a, "to_numpy") else a for a in args]
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


def text_enumeration(words: Sequence[Any], conjunction: str = "and", quotes: bool = False) -> str:
    if quotes:
        words = [f"'{word}'" for word in words]
    return f"{', '.join(words[:-1])} {conjunction} {words[-1]}"


T = TypeVar("T")


def check_input(value: T, valid_values: Sequence[Any]) -> T:
    if value not in valid_values:
        raise ValueError(f"invalid value '{value}' (must be {text_enumeration(valid_values, 'or', quotes=True)})")
    return value


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
