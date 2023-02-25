"""Defines BasisArray class, which allows the attachment of bases to NumPy arrays.
The arrays can then be used in a custom einsum function, where the overlaps between
the different bases are automatically taken into account.

Usage:

>>> root = Space(mol.nao, metric=mf.get_ovlp())
>>> mo = Space.add_basis(mf.mo_coeff, name='mo')
>>> mo_occ = Space.add_basis(mf.mo_coeff[:,occ], name='mo-occ')
>>> mo_vir = Space.add_basis(mf.mo_coeff[:,vir], name='mo-vir')
>>> fov = BasisArray(fock[occ,vir], basis=(mo_occ, mo_vir))
>>> # View in different basis:
>>> print(fov.as_basis((mo, mo)))
>>> # Contract with other BasisArray:
>>> result = basis_einsum('ia,ja->ij', fov, t1)
>>> # The virtual dimension of `fov` and `t1` can be expressed in a different basis;
>>> # The corresponding overlap will be considered automatically.
>>> # `result` is another `BasisArray` instance.

"""

import copy
import string
import functools
import uuid
import numpy as np
from util import *
from basis import BasisBase, RootBasis


class BasisArray:
    """NumPy array with bases attached for each dimension."""

    def __init__(self, value, basis, contravariant=False):
        if len(basis) != np.ndim(value):
            raise ValueError("Array with shape %r requires %d bases, %d given" % (
                value.shape, np.ndim(value), len(basis)))
        for i, b in enumerate(basis):
            if value.shape[i] != b.size:
                raise ValueError("Dimension %d with size %d incompatible with basis size %d" % (
                    i+1, value.shape[i], b.size))
        self.value = value
        self.basis = basis
        if np.ndim(contravariant) == 0:
            contravariant = self.ndim * [contravariant]
        self.contravariant = contravariant

    def __repr__(self):
        return '%s(shape= %r)' % (self.__class__.__name__, self.shape)

    def __getattr__(self, name):
        if name in ['dtype', 'ndim', 'shape', '__array_interface__']:
            return getattr(self.value, name)
        raise AttributeError("%r object has no attribute '%s'" % (self.__class__.__name__, name))

    def __getitem__(self, key):
        """Allow direct access of array data as array[key]."""
        return self.value[key]

    def as_basis(self, basis):
        """Transform to different set of bases.

        None can be used to indicate no transformation.

        Note that this can reduce the rank of the array, for example when trying to express
        a purely occupied quantitiy in a purely virtual basis.
        """
        if len(basis) != len(self.basis):
            raise ValueError
        for bas in basis:
            if not isinstance(bas, (BasisBase, type(None))):
                raise ValueError

        subscripts = string.ascii_lowercase[:self.ndim]
        operands = [self.value]
        result = ''
        basis_out = list(basis)
        for i, bas in enumerate(basis):
            if bas is None or (bas == self.basis[i]):
                result += subscripts[i]
                if bas is None:
                    basis_out[i] = self.basis[i]
                continue
            # If self.basis[i] is covariant and bas is contravariant (or vice versa), the order
            # of bases in the overlap matters:
            if not self.contravariant[i]:
                ovlp = (self.basis[i] | bas).value
            else:
                ovlp = (bas | self.basis[i]).value.T
            operands.append(ovlp)
            sub_new = subscripts[i].upper()
            subscripts += (',%s%s' % (subscripts[i], sub_new))
            result += sub_new
        subscripts += '->%s' % (''.join(result))
        value = np.einsum(subscripts, *operands, optimize=True)
        return BasisArray(value, tuple(basis_out), contravariant=self.contravariant)

    def __rshift__(self, basis):
        """To allow basis transformation as array >> basis"""
        return self.__or__(basis)

    def __or__(self, basis):
        """To allow basis transformation as (array | basis)"""
        if isinstance(basis, BasisBase):
            basis = (basis,)
        if isinstance(basis, tuple):
            basis = self.basis[:-len(basis)] + basis
        return self.as_basis(basis)

    def __rlshift__(self, basis):
        """To allow basis transformation as basis << array"""
        return self.__ror__(basis)

    def __ror__(self, basis):
        """To allow basis transformation as (basis | array)"""
        if isinstance(basis, BasisBase):
            basis = (basis,)
        if isinstance(basis, tuple):
            basis = basis + self.basis[len(basis):]
        return self.as_basis(basis)


def basis_einsum(subscripts, *operands, einsumfunc=np.einsum, **kwargs):
    """Allows contraction of BasisArray objects using Einstein notation.

    The overlap matrices between non-matching dimensions are automatically added.
    """
    # Remove spaces
    subscripts = subscripts.replace(' ', '')
    if '->' in subscripts:
        labels, result = subscripts.split('->')
    else:
        labels = subscripts
        result = ''
    labels = labels.split(',')
    assert (len(labels) == len(operands))

    labels = [list(label) for label in labels]
    labels_out = copy.deepcopy(labels)

    # Sorted set of all indices
    indices = [x for idx in labels for x in idx]
    # Remove duplicates while keeping order (sets do not keep order):
    indices = list(dict.fromkeys(indices).keys())
    # Unused indices
    free_indices = sorted(set(string.ascii_letters).difference(set(indices)))

    # Loop over all indices
    overlaps = []
    basis_dict = {}
    #contravariant = {}
    for idx in indices:

        # Find smallest basis for given idx:
        basis = None
        for i, label in enumerate(labels):
            # The index might appear multiple times per label -> loop over positions
            positions = np.asarray(np.asarray(label) == idx).nonzero()[0]
            for pos in positions:
                basis2 = operands[i].basis[pos]
                if basis is None or (basis2.size < basis.size):
                    basis = basis2
                #contra = operands[i].contravariant[pos]

        assert (basis is not None)
        basis_dict[idx] = basis
        #contravariant[idx] =

        # Replace all other bases corresponding to the same index:
        for i, label in enumerate(labels):
            positions = np.asarray(np.asarray(label) == idx).nonzero()[0]
            for pos in positions:
                basis2 = operands[i].basis[pos]
                # If the bases are the same, continue, to avoid inserting an unnecessary identity matrix:
                if basis2 == basis:
                    continue
                # Add transformation from basis2 to basis:
                idx2 = free_indices.pop(0)
                labels_out[i][pos] = idx2
                labels_out.append([idx, idx2])
                overlaps.append((basis | basis2).value)

    # Return
    subscripts_out = ','.join([''.join(label) for label in labels_out])
    subscripts_out = '->'.join((subscripts_out, result))
    operands_out = [op.value for op in operands]
    operands_out.extend(overlaps)
    values = einsumfunc(subscripts_out, *operands_out, **kwargs)
    basis_out = tuple([basis_dict[idx] for idx in result])

    return BasisArray(values, basis_out)


if __name__ == '__main__':

    import pyscf
    import pyscf.gto
    import pyscf.scf
    import pyscf.cc
    import scipy
    import scipy.stats

    mol = pyscf.gto.Mole()
    mol.atom = 'Li 0 0 0 ; Li 0 0 1.5'
    mol.basis = 'cc-pVDZ'
    mol.build()

    mf = pyscf.scf.HF(mol)
    mf.kernel()

    ccsd = pyscf.cc.CCSD(mf)
    ccsd.kernel()

    t2x = ccsd.t2
    mo_coeff_x = mf.mo_coeff
    occ = mf.mo_occ>0
    vir = mf.mo_occ==0
    mo_coeff_occ_x = mf.mo_coeff[:,mf.mo_occ>0]
    mo_coeff_vir_x = mf.mo_coeff[:,mf.mo_occ==0]

    # Random unitary rotation:
    u_occ = scipy.stats.ortho_group.rvs(mo_coeff_occ_x.shape[1])#[:,:2]
    u_vir = scipy.stats.ortho_group.rvs(mo_coeff_vir_x.shape[1])
    mo_coeff_occ_y = np.dot(mo_coeff_occ_x, u_occ)
    mo_coeff_vir_y = np.dot(mo_coeff_vir_x, u_vir)
    t2y = np.einsum('ijab,iI,jJ,aA,bB->IJAB', t2x, u_occ, u_occ, u_vir, u_vir)

    # Manual: pre-calculate overlaps:
    ovlp_ao = mf.get_ovlp()
    r_occ = np.linalg.multi_dot((mo_coeff_occ_x.T, ovlp_ao, mo_coeff_occ_y))
    r_vir = np.linalg.multi_dot((mo_coeff_vir_x.T, ovlp_ao, mo_coeff_vir_y))
    #result = np.einsum('ijab,IJAB,iI,jJ,aA->bB', t2x, t2y, r_occ, r_occ, r_vir)
    result = np.einsum('ijab,IJAB,iI,jJ,aA,bB->ij', t2x, t2y, r_occ, r_occ, r_vir, r_vir)
    #result = np.einsum('ijab,IJAB,iI,jJ,aA,bB->', t2x, t2y, r_occ, r_occ, r_vir, r_vir)
    #result = np.einsum('iiab,IJAB,iI,aA->bB', t2x, t2y, r_occ, r_vir)

    # Define roots and BasisArray instances:
    ao = root = RootBasis(mol.nao, metric=ovlp_ao)
    ao_sub = root.make_basis(indices=[0,1,2,3])
    mo = mo_x = root.make_basis(mo_coeff_x)
    indices = (mf.mo_occ>0).nonzero()[0]
    occ_x = mo_x.make_basis(indices=indices)
    indices = (mf.mo_occ==0).nonzero()[0]
    vir_x = mo_x.make_basis(indices=indices)
    occ_y = root.make_basis(mo_coeff_occ_y)
    vir_y = root.make_basis(mo_coeff_vir_y)

    t2x = BasisArray(t2x, (occ_x, occ_x, vir_x, vir_x), contravariant=True)
    t2y = BasisArray(t2y, (occ_y, occ_y, vir_y, vir_y), contravariant=True)
    #result2 = basis_einsum('ijab,ijaB->bB', t2x, t2y)
    result2 = basis_einsum('ijab,ijab->ij', t2x, t2y)
    #result2 = basis_einsum('iiab,ijaB->bB', t2x, t2y)
    #result2 = basis_einsum('ijab,ijab', t2x, t2y)

    c = (ao | occ_x)
    #print(np.linalg.norm(c - mo_coeff_occ_x))
    print(c.contravariant)

    #dm = c.as_basis((None, ao))
    dm = ((ao | occ_x) | ao)
    dm1 = mf.make_rdm1()
    print(np.linalg.norm(dm - dm1))
    #print(np.linalg.norm(dm - np.linalg.multi_dot((c.value))))

    1/0

    #ovlp = (c.basis[1] | ao)
    #print(ovlp.value.T - np.dot(ovlp_ao, mo_coeff_occ_x))

    ovlp = (ao | c.basis[1])
    print(ovlp.value - mo_coeff_occ_x)

    1/0

    # DM
    #dm = ((ao | occ_x)  | ao)
    dm = (ao | occ_x | ao)
    #dm1 = np.dot(ovlp_ao, dm1).dot(ovlp_ao)
    print(np.linalg.norm(dm - dm1))
    1/0


    assert np.allclose(result, result2.value)

    mat = np.random.rand(mol.nao, mol.nao)
    e, v = np.linalg.eigh(mat)
    c_frag = v[:,:3]

    frag = root.make_basis(c_frag)

    #print(ovlp)
    #print(ovlp.basis)

    print(ao.is_orthonormal)
    print(mo.is_orthonormal)

    ovlp = (ao|ao) # C_ai
    print(np.linalg.norm(ovlp - ovlp_ao))

    ovlp = (ao|mo) >> ao
    print(np.linalg.norm(ovlp - np.eye(ao.size)))

    ovlp = ao << (mo|ao)
    print(np.linalg.norm(ovlp - np.eye(ao.size)))

    ovlp = (ao|mo) >> ao
    print(np.linalg.norm(ovlp - np.eye(ao.size)))

    ovlp = (ao|mo) # C_ai

    print(np.linalg.norm(ovlp - mf.mo_coeff))
    #print(np.linalg.norm(ovlp - np.dot(ovlp_ao, mf.mo_coeff)))

    ovlp = (mo|ao) # C_ai * Sab
    #print(np.linalg.norm(ovlp - mf.mo_coeff.T))
    print(np.linalg.norm(ovlp - np.dot(ovlp_ao, mf.mo_coeff).T))

    ovlp = (mo|ao) >> mo
    print(np.linalg.norm(ovlp - np.eye(ao.size)))
    print(ovlp.contravariant)

    ovlp = mo << (ao|mo)
    print(np.linalg.norm(ovlp - np.eye(ao.size)))
    print(ovlp.contravariant)

    (mo|frag) >> mo

    hcore = mf.get_hcore()
    hcore_mo = np.linalg.multi_dot((mf.mo_coeff.T, hcore, mf.mo_coeff))

    h1e = BasisArray(hcore, (ao, ao))
    h1e_mo = BasisArray(hcore_mo, (mo, mo))

    test = (ao | h1e_mo | ao)
    test = h1e_mo >> (ao, ao)

    print(np.linalg.norm(test.value - h1e))
    print(test.contravariant)

    test = (mo | h1e | mo)
    print(np.linalg.norm(test.value - h1e_mo))
    print(test.contravariant)

    dm1_ao = mf.make_rdm1()
    dm1_mo = np.zeros_like(dm1_ao)
    nocc = np.count_nonzero(mf.mo_occ > 0)
    dm1_mo[np.diag_indices(nocc)] = 2

    bdm1_ao = BasisArray(dm1_ao, (ao, ao), contravariant=True)
    bdm1_mo = BasisArray(dm1_mo, (mo, mo), contravariant=True)

    dm1_half = np.einsum('ab,bc,ci->ai', dm1_ao, ovlp_ao, mf.mo_coeff)
    test = (bdm1_ao | mo)
    print(np.linalg.norm(test.value - dm1_half))

    test = (mo | bdm1_ao)
    print(np.linalg.norm(test.value - dm1_half.T))

    test = (ao | bdm1_mo | ao)
    print(np.linalg.norm(test.value - dm1_ao))

    test = (mo | bdm1_ao | mo)
    print(np.linalg.norm(test.value - dm1_mo))


    #ref = mf.mo_coeff[:,:3]

    #print(ovlp.shape)
    #print(ref.shape)
    #1/0

    #print('yyyy')
    #test = (occ_x | h1e_mo | vir_x)
    #print(test.value.shape)
    #print((hcore_mo[occ][:,vir]).shape)
    #print(np.linalg.norm(test.value - hcore_mo[occ][:,vir]))
    #1/0


    #ovlp = (mo_x | ao)
    #ovlp = (ao | mo_x)
    #ovlp2 = (mo_x | ao)

    #print(np.linalg.norm(ovlp.value - ovlp2.value.T))
    #1/0


    #print(type(ovlp))
    #print(ovlp.value)
    #print(np.linalg.norm(ovlp.value - np.dot(ovlp_ao, mo_x.coeff)))


    #1/0

    ovlp1 = (mo_x | frag) | mo_x
    ovlp2 = mo_x | (frag | mo_x)
    ovlp3 = (mo_x | frag | mo_x)

    proj = (mo | frag) >> (None, mo)

    #test = (occ_x, occ_x) | t2x | (vir_x, vir_x)
    #test = t2x | (occ_y, occ_y, vir_y, vir_y)
    #test = (occ_y, occ_y, vir_y, vir_y) | t2x
    #
    #test = (None, None) | t2x | (None, None)
    t2x_ao = (ao, ao) | t2x | (ao, ao)

    #r_occ = np.dot(ovlp_ao, mo_coeff_occ_x)
    #r_vir = np.dot(ovlp_ao, mo_coeff_vir_x)
    r_occ = mo_coeff_occ_x
    r_vir = mo_coeff_vir_x

    # ERIs are transformed without S, T is transformed with S!
    ref = np.einsum('ijab,pi,qj,ra,sb->pqrs', t2x, r_occ, r_occ, r_vir, r_vir, optimize=True)

    print(np.linalg.norm(t2x_ao.value - ref))

    t2x_mo = (occ_x, occ_x) | t2x_ao | (vir_x, vir_x)


    #t2x_mo = t2x_ao @ (vir_x, vir_x)


    print(np.linalg.norm(t2x_mo.value - t2x.value))

    print(ovlp1.shape)
    print(ovlp2.shape)

    print(np.linalg.norm(ovlp1.value - ovlp2.value))
    print(np.linalg.norm(ovlp1.value - ovlp3.value))

    #t_ij^ab * t_IJ^AB


    1/0

    # Test
    print(result.shape)
    print(result2.shape)
    print(np.linalg.norm(result - result2.value))

    print(result2.basis)
    assert np.allclose(result, result2)

    result3 = result2.as_basis((mo_x, mo_x))

    print(result3.shape)
    #print(result3.value)
    #assert np.allclose(result, result3)

    ovlp = (occ_y | occ_x)
    print(id(ovlp))

    ovlp = (occ_y | occ_x)
    print(id(ovlp))

    1/0

    #diff = (result2 - result)
    #print(np.linalg.norm(result - result2))
    #print(type(diff))
    print(result2.shape)
