from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy
import scipy.linalg

if TYPE_CHECKING:
    from btensor import Basis


class Space:

    def __init__(self, basis: Basis, svd_tol: float = 1e-12) -> None:
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
        #ovlp = (~self.basis | other.basis).to_numpy()
        ovlp = self.basis.get_transformation_to(other.basis).to_numpy()
        #ovlp = self.basis.get_overlap(other.basis, variance=[-1, -1]).to_numpy()
        sv = scipy.linalg.svd(ovlp, compute_uv=False)
        return sv

    def __eq__(self, other: Space) -> bool:
        """True, if self is the same space as other."""
        if not isinstance(other, Space):
            return False
        if len(self) != len(other):
            return False
        if not self.basis.same_root(other.basis):
            return False
        parent = self.basis.find_common_parent(other.basis)
        if len(parent) == len(self):
            return True
        # Perform SVD to determine relationship
        sv = self._singular_values_of_overlap(other)
        return np.all(abs(sv-1) < self._svd_tol)

    def __neq__(self, other: Space) -> bool:
        """True, if self is not the same space as other."""
        return not (self == other)

    def __lt__(self, other: Space) -> bool:
        """True, if self is a true subspace of other."""
        if not isinstance(other, Space):
            return False
        if len(self) >= len(other):
            return False
        if not self.basis.same_root(other.basis):
            return False
        if self.basis.is_derived_from(other.basis):
            return True
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

    def is_orthogonal(self, other: Space) -> bool:
        """True, if self is orthogonal to other."""
        if not isinstance(other, type(self)):
            raise TypeError
        if not self.basis.same_root(other.basis):
            return True
        parent_basis = self.basis.find_common_parent(other.basis)
        if parent_basis in (self, other):
            return False
        if len(self.basis) + len(other.basis) > len(parent_basis):
            return False
        sv = self._singular_values_of_overlap(other)
        return np.all(abs(sv) < self._svd_tol)

    def __or__(self, other: Space) -> bool:
        return self.is_orthogonal(other)
