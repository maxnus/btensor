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
from typing import TYPE_CHECKING

import numpy as np
import scipy
import scipy.linalg

if TYPE_CHECKING:
    from btensor import Basis


class Space:

    #: Default tolerance applied to eigendecompositions
    DEFAULT_TOL = 1e-12

    __doc__ = f"""A class describing the space spanned by some basis.
    
    Parameters
    ----------
    basis:
        Basis which spans the space.
    tol:
        Tolerance applied to eigenvalues of an eigendecomposition, in order to determine if a corresponding eigenvector
        is part of the space or not. Default: {DEFAULT_TOL}.
    """

    def __init__(self, basis: Basis, tol: float = DEFAULT_TOL) -> None:
        self._basis = basis
        self._tol = tol

    @property
    def basis(self) -> Basis:
        """Basis spanning the space."""
        return self._basis

    def __repr__(self) -> str:
        return f"{type(self).__name__}(basis= {self.basis}, size= {len(self)})"

    def __len__(self) -> int:
        return len(self.basis)

    def _singular_values_of_overlap(self, other: Space) -> np.ndarray:
        ovlp = self.basis.get_transformation_to(other.basis).to_numpy()
        sv = scipy.linalg.svd(ovlp, compute_uv=False)
        return sv

    def _eigenvalues_of_projector(self, other: Space) -> np.ndarray:
        ovlp = self.basis.get_overlap(other.basis).to_numpy()
        if other.basis.is_orthonormal:
            proj = np.dot(ovlp, ovlp.T)
        else:
            proj = np.linalg.multi_dot([ovlp, other.basis.metric.inverse.to_numpy(), ovlp.T])
        ev = scipy.linalg.eigh(proj, b=self.basis.metric.to_numpy())[0]
        return ev

    def trivially_equal(self, other: Space) -> bool | None:
        """Check if spaces are trivially equal (without performing SVD)"""
        if not isinstance(other, Space):
            raise TypeError(type(other))
        if len(self) != len(other):
            return False
        if not self.basis.same_root(other.basis):
            return False
        if len(self.basis.get_common_parent(other.basis)) == len(self):
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

    def trivially_less_or_equal_than(self, other: Space) -> bool | None:
        """Check if space is trivially a subspace (without performing SVD)"""
        if not isinstance(other, Space):
            raise TypeError(type(other))
        if len(self) > len(other):
            return False
        if not self.basis.same_root(other.basis):
            return False
        if self.basis.is_derived_from(other.basis, inclusive=True):
            return True
        return None

    def trivially_orthogonal(self, other: Space) -> bool | None:
        """Check if spaces are trivially orthogonal (without performing SVD)"""
        if not isinstance(other, Space):
            raise TypeError(type(other))
        if not self.basis.same_root(other.basis):
            return True
        parent_basis = self.basis.get_common_parent(other.basis)
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
        #sv = self._singular_values_of_overlap(other)
        ev = self._eigenvalues_of_projector(other)
        rv = np.all(abs(ev-1) < self._tol)
        return rv

    def __neq__(self, other: Space) -> bool:
        """True, if self is not the same space as other."""
        return not (self == other)

    def __lt__(self, other: Space) -> bool:
        """True, if self is a true subspace of other."""
        if (lt := self.trivially_less_than(other)) is not None:
            return lt
        # Perform SVD to determine relationship
        #sv = self._singular_values_of_overlap(other)
        ev = self._eigenvalues_of_projector(other)
        rv = np.all(abs(ev-1) < self._tol)
        #return np.all(ev > 1-self._tol)
        return rv

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
        return np.all(abs(sv) < self._tol)
