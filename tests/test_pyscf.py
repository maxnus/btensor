import unittest
import numpy as np
import scipy
import scipy.stats
from collections import namedtuple
import basis_array as basis
from testing import TestCase, rand_orth_mat


scf_data = namedtuple('scf_data', ('nao', 'nmo', 'nocc', 'mo_coeff', 'mo_energy', 'mo_occ', 'ovlp', 'fock', 'dm'))
cc_data = namedtuple('cc_data', ('dm',))


def make_scf_data(nao):
    np.random.seed(0)
    nao = nmo = 20
    nocc = 8
    nvir = nmo - nocc
    mo_coeff = rand_orth_mat(nao) + 0.1 * np.random.random((nao, nmo))
    mo_energy = np.random.uniform(-3.0, 3.0, nmo)
    mo_occ = np.asarray((nocc * [2] + nvir * [0]))
    sinv = np.dot(mo_coeff, mo_coeff.T)
    ovlp = np.linalg.inv(sinv)
    # Test biorthogonality
    csc = np.linalg.multi_dot((mo_coeff.T, ovlp, mo_coeff))
    assert np.allclose(csc-np.identity(nao), 0)
    fock = np.linalg.multi_dot((ovlp, mo_coeff, np.diag(mo_energy), mo_coeff.T, ovlp))
    dm = np.dot(mo_coeff * mo_occ[None], mo_coeff.T)
    return scf_data(nao, nmo, nocc, mo_coeff, mo_energy, mo_occ, ovlp, fock, dm)


def make_cc_data(scf):
    np.random.seed(100)
    dm = np.diag(scf.mo_occ) + np.random.uniform(-0.1, 0.1, (scf.nmo, scf.nmo))
    e, v = np.linalg.eigh(dm)
    e = np.clip(e, 0.0, 2.0)
    dm = np.dot(v*e[None], v.T)
    return cc_data(dm)


class SCF_Tests(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scf = make_scf_data(20)
        cls.cc = make_cc_data(cls.scf)
        cls.ao = basis.B(cls.scf.nao, metric=cls.scf.ovlp, name='AO')
        cls.mo = basis.B(cls.ao, rotation=cls.scf.mo_coeff, name='MO')

    def test_cc_dm_mo(self):
        nocc = sum(self.scf.mo_occ>0)
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]
        c_occ = self.scf.mo_coeff[:,occ]
        c_vir = self.scf.mo_coeff[:,vir]
        dm = self.cc.dm
        dm_oo = dm[occ,occ]
        dm_ov = dm[occ,vir]
        dm_vo = dm[vir,occ]
        dm_vv = dm[vir,vir]
        mo = basis.B(len(self.scf.mo_occ))
        bo = basis.B(mo, occ)
        bv = basis.B(mo, vir)
        bdm_oo = basis.A(dm_oo, basis=(bo, bo))
        bdm_ov = basis.A(dm_ov, basis=(bo, bv))
        bdm_vo = basis.A(dm_vo, basis=(bv, bo))
        bdm_vv = basis.A(dm_vv, basis=(bv, bv))
        bdm = (bdm_oo + bdm_ov + bdm_vo + bdm_vv)
        self.assertAllclose(bdm.value, dm)

    def test_ao_mo_transform(self):
        ao, mo = self.ao, self.mo
        self.assertAllclose((ao|ao), np.identity(self.scf.nao))
        self.assertAllclose((mo|mo), np.identity(self.scf.nao))
        self.assertAllclose((ao|mo), self.scf.mo_coeff)
        self.assertAllclose((mo|ao), np.dot(self.scf.mo_coeff.T, self.scf.ovlp))

    def test_ao_mo_projector(self):
        ao, mo = self.ao, self.mo
        csc = np.dot((mo|ao), (ao|mo))
        self.assertAllclose(csc, np.identity(self.scf.nao))
        csc = basis.dot((mo|ao), (ao|mo))
        self.assertAllclose(csc, np.identity(self.scf.nao))
        csc = np.dot((ao|mo), (mo|ao))
        self.assertAllclose(csc, np.identity(self.scf.nao))
        csc = basis.dot((ao|mo), (mo|ao))
        self.assertAllclose(csc, np.identity(self.scf.nao))

    def test_ao2mo_ovlp(self):
        ao, mo = self.ao, self.mo
        s = basis.A(self.scf.ovlp, basis=(ao, ao))
        self.assertAllclose(((mo|s)|mo), np.identity(self.scf.nao))
        self.assertAllclose((mo|(s|mo)), np.identity(self.scf.nao))

    def test_mo2ao_ovlp(self):
        ao, mo = self.ao, self.mo
        s = basis.A(np.identity(self.scf.nao), basis=(mo, mo))
        self.assertAllclose(((ao|s)|ao), self.scf.ovlp)
        self.assertAllclose((ao|(s|ao)), self.scf.ovlp)

    def test_ao2mo_fock(self):
        ao, mo = self.ao, self.mo
        f = basis.A(self.scf.fock, basis=(ao, ao))
        self.assertAllclose(((mo|f)|mo), np.diag(self.scf.mo_energy), atol=1e-9)
        self.assertAllclose((mo|(f|mo)), np.diag(self.scf.mo_energy), atol=1e-9)

    def test_mo2ao_fock(self):
        ao, mo = self.ao, self.mo
        f = basis.A(np.diag(self.scf.mo_energy), basis=(mo, mo))
        self.assertAllclose(((ao|f)|ao), self.scf.fock, atol=1e-9)
        self.assertAllclose((ao|(f|ao)), self.scf.fock, atol=1e-9)

    def test_ao2mo_dm(self):
        ao, mo = self.ao, self.mo
        d = basis.A(self.scf.dm, basis=(ao, ao), variance=(-1, -1))
        self.assertAllclose(((mo|d)|mo), np.diag(self.scf.mo_occ))
        self.assertAllclose((mo|(d|mo)), np.diag(self.scf.mo_occ))

    def test_mo2ao_dm(self):
        ao, mo = self.ao, self.mo
        d = basis.A(np.diag(self.scf.mo_occ), basis=(mo, mo), variance=(-1, -1))
        self.assertAllclose(((ao|d)|ao), self.scf.dm)
        self.assertAllclose((ao|(d|ao)), self.scf.dm)


    def test_1(self):
        return
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
        ao = root = basis.RootBasis(mol.nao, metric=ovlp_ao)
        ao_sub = root.make_basis(indices=[0,1,2,3])
        mo = mo_x = root.make_basis(mo_coeff_x)
        indices = (mf.mo_occ>0).nonzero()[0]
        occ_x = mo_x.make_basis(indices=indices)
        indices = (mf.mo_occ==0).nonzero()[0]
        vir_x = mo_x.make_basis(indices=indices)
        occ_y = root.make_basis(mo_coeff_occ_y)
        vir_y = root.make_basis(mo_coeff_vir_y)

        t2x = basis.Array(t2x, (occ_x, occ_x, vir_x, vir_x), contravariant=True)
        t2y = basis.Array(t2y, (occ_y, occ_y, vir_y, vir_y), contravariant=True)
        #result2 = basis_einsum('ijab,ijaB->bB', t2x, t2y)
        result2 = basis.einsum('ijab,ijab->ij', t2x, t2y)
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

        h1e = basis.Array(hcore, (ao, ao))
        h1e_mo = basis.Array(hcore_mo, (mo, mo))

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

        bdm1_ao = basis.Array(dm1_ao, (ao, ao), contravariant=True)
        bdm1_mo = basis.Array(dm1_mo, (mo, mo), contravariant=True)

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


if __name__ == '__main__':
    unittest.main()
