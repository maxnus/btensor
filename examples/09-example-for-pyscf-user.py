import numpy as np
import scipy
import scipy.stats

import btensor

# PySCF scenario:

import pyscf
import pyscf.cc

mol = pyscf.gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.7'
mol.basis = '6-31G'
mol.build()

hf = pyscf.scf.HF(mol)
hf.kernel()

cc = pyscf.cc.CCSD(hf)
cc.kernel()
print()

# BTensor


# Set up AO root-basis:
ao = btensor.Basis(mol.nao, metric=hf.get_ovlp())


# Define tensor for HF density and Fock matrices (note that the Hcore/Fock matrix transform covariantly):
dm = btensor.Tensor(hf.make_rdm1(), basis=(ao, ao))
hcore = btensor.Cotensor(hf.get_hcore(), basis=(ao, ao))
fock = btensor.Cotensor(hf.get_fock(), basis=(ao, ao))
# Alternatively, the variance can be specified along each dimension: covariant = 1, contravariant = -1:
# fock = btensor.Tensor(hf.get_fock(), basis=(ao, ao), variance=(1, 1))


# By setting orthonormal to true, we avoid using the MO metric C^T.S.C to raise or lower tensor indices:
mo = ao.make_subbasis(hf.mo_coeff, orthonormal=True)


# The HF matrices can be transformed to the MO basis, using [ ]:
print("HF density-matrix in MO basis:")
print(dm[mo, mo].to_numpy())
print()
print("HF Fock-matrix in MO basis:")
print(fock[mo, mo].to_numpy())
print()


# In addition to defining a sub-basis in terms of a general 2D transformation matrix (such was done for the MOs above),
# in cases where no transformation is desired, sub-bases can also be defined in terms of index
# sequences (using integers), slices, or masking sequences (using bools):
nocc = np.count_nonzero(hf.mo_occ > 0)
nvir = len(hf.mo_occ) - nocc
occ = mo.make_subbasis(list(range(nocc)), orthonormal=True)
vir = mo.make_subbasis(slice(nocc, None), orthonormal=True)
act = mo.make_subbasis([True, True, False, False], orthonormal=True)


# If the basis in [ ] is a sub-basis, a projection is performed automatically:
print("Occupied-occupied block of HF density-matrix:")
print(dm[occ, occ].to_numpy())
print()
print("Virtual-virtual block of HF density-matrix:")
print(dm[vir, vir].to_numpy())
print()
print("HF Fock-matrix in active space:")
print(fock[act, act].to_numpy())
print()


# To perform a pure change of basis and raise an exception, if a projection would occur, the .cob (change of basis)
# interface can be used.
dm.cob[mo, mo]           # OK - pure basis transformation
try:
    dm.cob[occ, occ]     # Not OK - transformation and projection onto occupied subspace
except Exception as e:
    print(f"Exception encountered:\n{e.__class__.__name__}: {e}\n")


# Scalar operations work as expected:
error = np.linalg.norm((2*fock).to_numpy() - 2*(fock.to_numpy()))
print(f"Scalar multiplication error = {error}\n")


# Tensors can be added or subtracted, even if they have different bases, as long as they are compatible.
# Two tensors are compatible, if they have the same number of dimensions and the bases along each axis have the same
# root-basis (or are the root-basis itself):
fock_vir = fock[vir, vir]
fock_occ = fock - fock_vir
print("Fock-matrix without virtual-virtual part in MO basis:")
print(fock_occ[mo, mo].to_numpy())
print()


# This behavior can be used to assemble a tensor from multiple sub-tensors within a larger space:
dm_cc = btensor.Tensor(cc.make_rdm1(), basis=(mo, mo))
dm_oo = dm_cc[occ, occ]
dm_ov = dm_cc[occ, vir]
dm_vv = dm_cc[vir, vir]
dm_cc_reassembled = dm_oo + dm_ov + dm_ov.T + dm_vv     # Note the direct addition between the tensors
error = np.linalg.norm(dm_cc.to_numpy() - dm_cc_reassembled.to_numpy())
print(f"Error in reassembled matrix = {error}\n")


# All other elementwise operation (except for addition and subtraction), e.g. elementwise multiplication,
# are only permitted, if both tensors have the same basis along each axis. The reason for this is that the result of
# these operations depends on the basis that they are carried out in, hence it is required to be explicit about this
# choice:
dm_oo[mo, mo] * dm_vv[mo, mo]         # OK: same basis
dm_oo[mo, mo] + dm_vv[ao, ao]         # also OK: addition
try:
    dm_oo[mo, mo] * dm_vv[ao, ao]     # not OK
except Exception as e:
    print(f"Exception encountered:\n{e.__class__.__name__}: {e}\n")


# Note that the value of the trace is correct, even though dm was defined in the AO basis.
# The metric is inserted automatically to obtain the mixed covariant-contravariant representation.
print(f"Trace of density-matrix: {dm.trace()}\n")


# Intersection and union basis can be constructed:
inter = vir.make_intersect_basis(act)
union = vir.make_union_basis(act)
print(f"Active basis size:       {len(act)}")
print(f"Virtual basis size:      {len(vir)}")
print(f"Intersection basis size: {len(inter)}")
print(f"Union basis size:        {len(union)}\n")


# The function btensor.einsum can contract tensors defined in different bases.
# Assuming two differnt sets of virtual orbitals and correspondings T2 amplitudes:
nvir_x = nvir_y = 2
vir_x = vir.make_subbasis(scipy.stats.ortho_group.rvs(nvir)[:, :nvir_x])
vir_y = vir.make_subbasis(scipy.stats.ortho_group.rvs(nvir)[:, :nvir_y])
x2y = vir_x.get_transformation_to(vir_y).to_numpy()
t2x = btensor.Tensor(np.random.random((nocc, nocc, nvir_x, nvir_y)), basis=(occ, occ, vir_x, vir_x))
t2y = btensor.Tensor(np.random.random((nocc, nocc, nvir_x, nvir_y)), basis=(occ, occ, vir_y, vir_y))
# Usually a contraction would look like this:
expected = np.einsum('ijab,kjAB,aA,bB->ik', t2x.to_numpy(), t2y.to_numpy(), x2y, x2y)
# Using btensor.einsum, the basis transformation matrices are not necessary:
result = btensor.einsum('ijab,kjab->ik', t2x, t2y)
error = np.linalg.norm(result.to_numpy() - expected)
print(f"Error of btensor.einsum = {error}\n")

