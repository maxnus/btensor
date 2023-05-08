import numpy as np
from btensor import Basis


basis = Basis(3)

# --- A derived basis can be constructed in terms of
# 1) A general transformation matrix
tfm = np.asarray([[1, 0],
                  [0, 0],
                  [0, 1]])
basis1 = Basis(tfm, parent=basis)
# 2) an indexing array
basis2 = Basis([0, 2], parent=basis)
# 3) a slice object
basis3 = Basis(slice(0, 3, 2), parent=basis)
# 4) a masking array
basis4 = Basis([True, False, True], parent=basis)

# --- Note that all the definitions above are equivalent:
assert np.all(basis1.coeff_in_basis(basis).evaluate() == tfm)
assert np.all(basis2.coeff_in_basis(basis).evaluate() == tfm)
assert np.all(basis3.coeff_in_basis(basis).evaluate() == tfm)
assert np.all(basis4.coeff_in_basis(basis).evaluate() == tfm)
