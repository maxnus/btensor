from __future__ import annotations
from typing import Self, Any, TYPE_CHECKING

import numpy as np
import scipy
import scipy.linalg

if TYPE_CHECKING:
    from btensor import Basis


class Space:

    def __init__(self, basis: Basis, svd_tol: float = 1e-12):
        self._basis = basis
        self._svd_tol = svd_tol

    @property
    def basis(self) -> Basis:
        return self._basis

    def __repr__(self) -> str:
        return f"{type(self).__name__}(basis= {self.basis})"

    def __len__(self) -> int:
        return len(self.basis)

    def _svd(self, other: Self) -> np.ndarray:
        ovlp = (~self.basis | other.basis).to_numpy()
        sv = scipy.linalg.svd(ovlp, compute_uv=False)
        return sv

    def __eq__(self, other: Any) -> bool:
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
        sv = self._svd(other)
        return np.all(abs(sv-1) < self._svd_tol)

    def __neq__(self, other: Any) -> bool:
        """True, if self is not the same space as other."""
        return not (self == other)

    def __lt__(self, other: Any) -> bool:
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
        sv = self._svd(other)
        return np.all(sv > 1-self._svd_tol)

    def __le__(self, other: Any) -> bool:
        """True, if self a subspace of other or spans the same space."""
        return (self < other) or (self == other)

    def __gt__(self, other: Any) -> bool:
        """True, if self is a true superspace of other."""
        return other < self

    def __ge__(self, other: Any) -> bool:
        """True, if self a superspace of other or spans the same space."""
        return (self > other) or (self == other)

    def __or__(self, other: Self) -> bool:
        """True, if self is orthogonal to other."""
        if not isinstance(other, type(self)):
            raise TypeError
        if not self.basis.same_root(other.basis):
            return True
        if self in other.basis.get_parents() or other in self.basis.get_parents():
            return True
        sv = self._svd(other)
        return np.all(abs(sv) < self._svd_tol)
