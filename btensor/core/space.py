from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy
import scipy.linalg

if TYPE_CHECKING:
    from btensor import Basis


class Space:

    DEFAULT_SVD_TOL = 1e-12

    def __init__(self, basis: Basis, svd_tol: float = DEFAULT_SVD_TOL) -> None:
        self._basis = basis
        self._svd_tol = svd_tol

    @property
    def basis(self) -> Basis:
        return self._basis

    def __repr__(self) -> str:
        return f"{type(self).__name__}(basis= {self.basis})"

    def __len__(self) -> int:
        return len(self.basis)

    def _singular_values_of_overlap(self, other: Space) -> np.ndarray:
        ovlp = self.basis.get_transformation_to(other.basis).to_numpy()
        sv = scipy.linalg.svd(ovlp, compute_uv=False)
        return sv

    def trivially_equal(self, other: Space) -> bool | None:
        """Check if spaces are trivially equal (without performing SVD)"""
        if not isinstance(other, Space):
            raise TypeError(type(other))
        if len(self) != len(other):
            return False
        if not self.basis.same_root(other.basis):
            return False
        parent = self.basis.find_common_parent(other.basis)
        if len(parent) == len(self):
            return True
        return None

    def trivially_less_than(self, other: Space) -> bool | None:
        """Check if space is trivially a true subspace (without performing SVD)"""
        if not isinstance(other, Space):
            raise TypeError(type(other))
        if len(self) >= len(other):
            return False
        if not self.basis.same_root(other.basis):
            return False
        if self.basis.is_derived_from(other.basis):
            return True
        return None

    def trivially_orthogonal(self, other: Space) -> bool | None:
        """Check if spaces are trivially orthogonal (without performing SVD)"""
        if not isinstance(other, Space):
            raise TypeError(type(other))
        if not self.basis.same_root(other.basis):
            return True
        parent_basis = self.basis.find_common_parent(other.basis)
        if parent_basis == self.basis or parent_basis == other.basis:
            return False
        if len(self.basis) + len(other.basis) > len(parent_basis):
            return False
        return None

    def __eq__(self, other: Space) -> bool:
        """True, if self is the same space as other."""
        if (eq := self.trivially_equal(other)) is not None:
            return eq
        # Perform SVD to determine relationship
        sv = self._singular_values_of_overlap(other)
        return np.all(abs(sv-1) < self._svd_tol)

    def __neq__(self, other: Space) -> bool:
        """True, if self is not the same space as other."""
        return not (self == other)

    def __lt__(self, other: Space) -> bool:
        """True, if self is a true subspace of other."""
        if (lt := self.trivially_less_than(other)) is not None:
            return lt
        # Perform SVD to determine relationship
        sv = self._singular_values_of_overlap(other)
        return np.all(sv > 1-self._svd_tol)

    def __le__(self, other: Space) -> bool:
        """True, if self a subspace of other or spans the same space."""
        return (self < other) or (self == other)

    def __gt__(self, other: Space) -> bool:
        """True, if self is a true superspace of other."""
        return other < self

    def __ge__(self, other: Space) -> bool:
        """True, if self a superspace of other or spans the same space."""
        return (self > other) or (self == other)

    def __or__(self, other: Space) -> bool:
        """True, if self is orthogonal to other."""
        if (orth := self.trivially_orthogonal(other)) is not None:
            return orth
        sv = self._singular_values_of_overlap(other)
        return np.all(abs(sv) < self._svd_tol)
