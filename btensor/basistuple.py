from typing import Optional, Self

from btensor.basis import BasisType, nobasis, compatible_basis, is_nobasis, find_common_parent


class BasisTuple:

    def __init__(self, basis: BasisType | tuple[BasisType, ...]):
        if isinstance(basis, BasisType):
            basis = (basis,)
        for bas in basis:
            if not isinstance(bas, BasisType):
                raise TypeError(f"type {type(BasisType)} required, not {type(bas)}")
        self._basis = basis

    @property
    def shape(self) -> tuple[Optional[int], ...]:
        return tuple(getattr(basis, 'size', None) for basis in self)

    def __len__(self) -> int:
        return len(self._basis)

    def __getitem__(self, key) -> BasisType | Self:
        if isinstance(key, slice):
            return BasisTuple(self._basis[key])
        return self._basis[key]

    def get_basistuple(self) -> tuple[BasisType, ...]:
        return self._basis

    def get_hashtuple(self) -> tuple[int, ...]:
        return tuple(hash(basis) for basis in self)

    def __eq__(self, other: Self) -> bool:
        return isinstance(other, BasisTuple) and self.get_hashtuple() == other.get_hashtuple()

    def is_compatible_with(self, other: Self) -> bool:
        if len(self) != len(other):
            return False
        for bas_self, bas_other in zip(self, other):
            if not compatible_basis(bas_self, bas_other):
                return False
        return True

    def get_common_basistuple(self, other: Self) -> Self:
        if not self.is_compatible_with(other):
            raise ValueError
        common_parents = tuple(find_common_parent(basis_self, basis_other)
                               for (basis_self, basis_other) in zip(self, other))
        return type(self)(common_parents)

    def update_with(self, update: tuple[Optional[BasisType]], check_size: bool = True) -> Self:
        new_basis = list(self.get_basistuple())
        if len(update) > len(self):
            raise ValueError
        for axis, (size, b0, b1) in enumerate(zip(self.shape, self, update)):
            if b1 is None:
                continue
            if check_size and not is_nobasis(b1) and b1.size != size:
                raise ValueError(f"axis {axis} with size {size} incompatible with basis size {b1.size}")
            new_basis[axis] = b1
        return BasisTuple(tuple(new_basis))

    # ---

    @property
    def variance(self) -> tuple[int]:
        return tuple(basis.variance for basis in self)

    @property
    def variance_string(self) -> str:
        """String representation of variance tuple."""
        symbols = {1: '+', -1: '-', 0: '*'}
        return ''.join(symbols[x] for x in self.variance)

    def is_spanning(self, other: Self) -> bool:
        for basis_self, basis_other in zip(self, other):
            if is_nobasis(basis_other):
                continue
            if is_nobasis(basis_self):
                return False
            basis_self = basis_self.get_nondual()
            basis_other = basis_other.get_nondual()
            if basis_other.space > basis_self.space:
                return False
        return True
