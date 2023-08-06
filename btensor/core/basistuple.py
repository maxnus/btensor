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
from typing import *
try:
    from types import EllipsisType
except ImportError:
    EllipsisType = type(Ellipsis)

from .basis import BasisInterface, compatible_basis, is_nobasis, get_common_parent, TBasis


KeyLike: TypeAlias = Union[BasisInterface, slice, EllipsisType]


class BasisTuple(tuple):

    def __init__(self, args: TBasis) -> None:
        for arg in args:
            if not isinstance(arg, BasisInterface):
                raise TypeError(f"{type(self).__name__} can only contain elements of type {BasisInterface.__name__} "
                                f"(not {arg})")

    @classmethod
    def create(cls, basis: TBasis) -> Self:
        if isinstance(basis, cls):
            return basis
        if not isinstance(basis, tuple):
            basis = (basis,)
        return cls(basis)

    @classmethod
    def create_from_default(cls,
                            basis: KeyLike | tuple[KeyLike, ...],
                            default: Self,
                            leftpad: bool = False) -> Self:
        if basis == slice(None) or basis == Ellipsis:
            return default
        if isinstance(basis, slice):
            raise ValueError(f"Only slice(None) is accepted (given: {basis}")
        if not isinstance(basis, tuple):
            basis = (basis,)

        nmissing = len(default) - len(basis)
        if nmissing < 0:
            raise ValueError(f"basis tuple with size {len(basis)} is larger than default with size {len(default)}")
        if nmissing > 0 and Ellipsis not in basis:
            if leftpad:
                basis = (Ellipsis,) + basis
            else:
                basis += (Ellipsis,)
        if Ellipsis in basis:
            idx = basis.index(Ellipsis)
            basis = basis[:idx] + nmissing*(slice(None),) + basis[idx+1:]
        if len(basis) != len(default):
            raise RuntimeError

        basis = [b1 if b1 != slice(None) else b0 for b1, b0 in zip(basis, default)]
        #for bas in basis:
        #    if not isinstance(bas, BasisInterface):
        #        raise TypeError(f"type {BasisInterface} required, not {type(bas)}")
        return cls(basis)

    @property
    def shape(self) -> tuple[Optional[int], ...]:
        return tuple(getattr(basis, 'size', None) for basis in self)

    @overload
    def __getitem__(self, key: int) -> BasisInterface: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    def __getitem__(self, key: int | slice) -> BasisInterface | Self:
        result = super().__getitem__(key)
        if isinstance(result, tuple):
            return type(self)(result)
        return result

    def is_compatible_with(self, other: Self) -> bool:
        if len(self) != len(other):
            return False
        for bas_self, bas_other in zip(self, other):
            if not compatible_basis(bas_self, bas_other):
                return False
        return True

    def get_root_basistuple(self) -> Self:
        return type(self)(basis.root for basis in self)

    def get_common_basistuple(self, other: Self) -> Self:
        if not self.is_compatible_with(other):
            raise ValueError
        common_parents = tuple(get_common_parent(basis_self, basis_other)
                               for (basis_self, basis_other) in zip(self, other))
        return type(self)(common_parents)

    def update_with(self, update: tuple[Optional[BasisInterface]], check_size: bool = True) -> Self:
        new_basis = list(self)
        if len(update) > len(self):
            raise ValueError
        for axis, (size, b0, b1) in enumerate(zip(self.shape, self, update)):
            if b1 is None:
                continue
            if check_size and not is_nobasis(b1) and b1.size != size:
                raise ValueError(f"axis {axis} with size {size} incompatible with basis size {b1.size}")
            new_basis[axis] = b1
        return BasisTuple.create(tuple(new_basis))

    # ---

    def is_spanning(self, other: Self) -> bool:
        for basis_self, basis_other in zip(self, other):
            if is_nobasis(basis_other):
                continue
            if is_nobasis(basis_self):
                return False
            if basis_other.space > basis_self.space:
                return False
        return True

