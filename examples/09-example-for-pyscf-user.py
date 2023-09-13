import numpy as np

import btensor

# PySCF scenario:

import pyscf

mol = pyscf.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.7'
mol.basis = '6-31G'
mol.build()

hf = pyscf.scf.HF(mol)
hf.kernel()

cc = pyscf.cc.CCSD(hf)
cc.kernel()

# BTensor

# Set up AO root-basis:
ao = btensor.Basis(mol.nao, metric=hf.get_ovlp())

# Define tensor for HF density and Fock matrices (note that the Hcore/Fock matrix transform covariantly):
dm_hf = btensor.Tensor(hf.make_rdm1(), basis=(ao, ao))
hcore = btensor.Cotensor(hf.get_hcore(), basis=(ao, ao))
fock = btensor.Cotensor(hf.get_fock(), basis=(ao, ao))
# Alternatively, the variance can be specified along each dimension: covariant = 1, contravariant = -1:
# fock = btensor.Tensor(hf.get_fock(), basis=(ao, ao), variance=(1, 1))

# By setting orthonormal to true, we avoid using the MO metric C^T.S.C to raise or lower tensor indices:
mo = ao.make_subbasis(hf.mo_coeff, orthonormal=True)

# The HF matrices can be transformed to the MO basis, using []:
print("HF density-matrix in MO basis:")
print(dm_hf[mo, mo].to_numpy())
print("HF Fock-matrix in MO basis:")
print(fock[mo, mo].to_numpy())

# Sub-bases can be defined in terms of slices, indexing (using integers), or masking (using bools) sequences:
nocc = np.count_nonzero(hf.mo_occ > 0)
occ = mo.make_subbasis(list(range(nocc)), orthonormal=True)
vir = mo.make_subbasis(slice(nocc, None), orthonormal=True)
act = mo.make_subbasis([True, True] + 4*[False], orthonormal=True)

# If the basis in [ ] is a sub-basis, a projection is performed automatically:
print("Occupied-occupied block of HF density-matrix:")
print(dm_hf[occ, occ].to_numpy())
print("HF Fock-matrix in active space:")
print(fock[act, act].to_numpy())

# To perform a pure change of basis and raise an exception, if a projection would occur, the .cob (change of basis)
# interface can be used.
dm_hf.cob[mo, mo]           # OK - pure basis transformation
try:
    dm_hf.cob[occ, occ]     # Not OK - transformation and projection onto occupied subspace
except Exception as e:
    print(f"Exception encountered: {e}")

# Tensors can be added, even if they have different bases, as long as they are compatible.
# Two tensors are compatible, if they have the same number of dimensions and the bases along each axis have the same
# root-basis (or are the root-basis itself):
fock_vir = fock[vir, vir]
f1 = fock - fock_vir
f2 = -fock_vir + fock
print(f1.basis)
print(f2.basis)














