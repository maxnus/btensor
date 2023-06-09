from typing import Optional, Self

from .basis import BasisType, compatible_basis, is_nobasis, find_common_parent


class BasisTuple(tuple):

    @classmethod
    def create(cls, basis: BasisType | tuple[BasisType, ...] | Self) -> Self:
        if isinstance(basis, cls):
            return basis
        if not isinstance(basis, tuple):
            basis = (basis,)
        for bas in basis:
            if not isinstance(bas, BasisType):
                raise TypeError(f"type {BasisType} required, not {type(bas)}")
        return cls(basis)

    @classmethod
    def create_with_default(cls,
                            basis: BasisType | None | tuple[BasisType | None, ...] | Self,
                            default: Optional[Self] = None) -> Self:
        if isinstance(basis, cls):
            return basis
        if not isinstance(basis, tuple):
            basis = (basis,)
        if default:
            basis = [b1 if b1 is not None else b0 for (b1, b0) in zip(basis, default)]
        for bas in basis:
            if not isinstance(bas, BasisType):
                raise TypeError(f"type {BasisType} required, not {type(bas)}")
        return cls(basis)

    @property
    def shape(self) -> tuple[Optional[int], ...]:
        return tuple(getattr(basis, 'size', None) for basis in self)

    def __getitem__(self, key) -> BasisType | Self:
        result = super().__getitem__(key)
        if isinstance(result, tuple):
            return type(self)(result)
        return result

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

    def get_root_basistuple(self) -> Self:
        return type(self)(basis.root for basis in self)

    def get_common_basistuple(self, other: Self) -> Self:
        if not self.is_compatible_with(other):
            raise ValueError
        common_parents = tuple(find_common_parent(basis_self, basis_other)
                               for (basis_self, basis_other) in zip(self, other))
        return type(self)(common_parents)

    def update_with(self, update: tuple[Optional[BasisType]], check_size: bool = True) -> Self:
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

