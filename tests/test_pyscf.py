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
        cls.mo = basis.B(cls.scf.mo_coeff, parent=cls.ao, name='MO')

    def test_cc_dm_mo(self):
        nocc = self.scf.nocc
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]
        dm = self.cc.dm
        dm_oo = dm[occ, occ]
        dm_ov = dm[occ, vir]
        dm_vo = dm[vir, occ]
        dm_vv = dm[vir, vir]
        mo = basis.B(len(self.scf.mo_occ))
        #occ = np.arange(self.scf.nmo)[occ]
        #vir = np.arange(self.scf.nmo)[vir]
        bo = basis.B(occ, parent=mo)
        bv = basis.B(vir, parent=mo)
        bdm_oo = basis.A(dm_oo, basis=(bo, bo))
        bdm_ov = basis.A(dm_ov, basis=(bo, bv))
        bdm_vo = basis.A(dm_vo, basis=(bv, bo))
        bdm_vv = basis.A(dm_vv, basis=(bv, bv))
        bdm = (bdm_oo + bdm_ov + bdm_vo + bdm_vv)
        self.assertAllclose(bdm.value, dm)

    def test_ao_mo_transform(self):
        ao, mo = self.ao, self.mo
        c = self.scf.mo_coeff
        s = self.scf.ovlp
        i = np.identity(self.scf.nao)

        # Normal-normal
        self.assertAllclose((ao | ao), s)
        self.assertAllclose((mo | mo), i)
        self.assertAllclose((ao | mo), s.dot(c))
        self.assertAllclose((mo | ao), c.T.dot(s.T))
        # Dual-normal
        self.assertAllclose((~ao | ao), i)
        self.assertAllclose((~mo | mo), i)
        self.assertAllclose((~ao | mo), c)
        self.assertAllclose((~mo | ao), c.T.dot(s.T))
        # Normal-dual
        self.assertAllclose((ao | ~ao), i)
        self.assertAllclose((mo | ~mo), i)
        self.assertAllclose((ao | ~mo), s.dot(c))
        self.assertAllclose((mo | ~ao), c.T)
        # Dual-dual
        self.assertAllclose((~ao | ~ao), np.linalg.inv(s))
        self.assertAllclose((~mo | ~mo), i)
        self.assertAllclose((~ao | ~mo), c)
        self.assertAllclose((~mo | ~ao), c.T)

    def test_ao_mo_projector(self):
        ao, mo = self.ao, self.mo
        i = np.identity(self.scf.nao)
        # NumPy
        # 1
        self.assertAllclose(np.dot((~ao| mo), ( mo|ao)), i)
        self.assertAllclose(np.dot((~ao|~mo), ( mo|ao)), i)
        self.assertAllclose(np.dot((~ao| mo), (~mo|ao)), i)
        self.assertAllclose(np.dot((~ao|~mo), (~mo|ao)), i)
        # 2
        self.assertAllclose(np.dot(( mo|~ao), (ao| mo)), i)
        self.assertAllclose(np.dot((~mo|~ao), (ao| mo)), i)
        self.assertAllclose(np.dot(( mo|~ao), (ao|~mo)), i)
        self.assertAllclose(np.dot((~mo|~ao), (ao|~mo)), i)
        # 3
        self.assertAllclose(np.dot(( mo|ao), (~ao| mo)), i)
        self.assertAllclose(np.dot((~mo|ao), (~ao| mo)), i)
        self.assertAllclose(np.dot(( mo|ao), (~ao|~mo)), i)
        self.assertAllclose(np.dot((~mo|ao), (~ao|~mo)), i)
        # 4
        self.assertAllclose(np.dot((ao| mo), ( mo|~ao)), i)
        self.assertAllclose(np.dot((ao|~mo), ( mo|~ao)), i)
        self.assertAllclose(np.dot((ao| mo), (~mo|~ao)), i)
        self.assertAllclose(np.dot((ao|~mo), (~mo|~ao)), i)

        # basis
        self.assertAllclose(basis.dot((~ao| mo), ( mo|ao)), i)
        self.assertAllclose(basis.dot((~ao|~mo), ( mo|ao)), i)
        self.assertAllclose(basis.dot((~ao| mo), (~mo|ao)), i)
        self.assertAllclose(basis.dot((~ao|~mo), (~mo|ao)), i)
        # 2
        self.assertAllclose(basis.dot(( mo|~ao), (ao| mo)), i)
        self.assertAllclose(basis.dot((~mo|~ao), (ao| mo)), i)
        self.assertAllclose(basis.dot(( mo|~ao), (ao|~mo)), i)
        self.assertAllclose(basis.dot((~mo|~ao), (ao|~mo)), i)
        # 3
        self.assertAllclose(basis.dot(( mo|ao), (~ao| mo)), i)
        self.assertAllclose(basis.dot((~mo|ao), (~ao| mo)), i)
        self.assertAllclose(basis.dot(( mo|ao), (~ao|~mo)), i)
        self.assertAllclose(basis.dot((~mo|ao), (~ao|~mo)), i)
        # 4
        self.assertAllclose(basis.dot((ao| mo), ( mo|~ao)), i)
        self.assertAllclose(basis.dot((ao|~mo), ( mo|~ao)), i)
        self.assertAllclose(basis.dot((ao| mo), (~mo|~ao)), i)
        self.assertAllclose(basis.dot((ao|~mo), (~mo|~ao)), i)

    def test_ao2mo_ovlp(self):
        ao, mo = self.ao, self.mo
        s = basis.A(self.scf.ovlp, basis=(ao, ao))
        self.assertAllclose(((mo | s) | mo), np.identity(self.scf.nao))
        self.assertAllclose((mo | (s | mo)), np.identity(self.scf.nao))

    def test_mo2ao_ovlp(self):
        ao, mo = self.ao, self.mo
        s = basis.A(np.identity(self.scf.nao), basis=(mo, mo))
        self.assertAllclose(((ao | s) | ao), self.scf.ovlp)
        self.assertAllclose((ao | (s | ao)), self.scf.ovlp)

    def test_ao2mo_fock(self):
        ao, mo = self.ao, self.mo
        f = basis.A(self.scf.fock, basis=(ao, ao))
        self.assertAllclose(((mo | f) | mo), np.diag(self.scf.mo_energy), atol=1e-9)
        self.assertAllclose((mo | (f | mo)), np.diag(self.scf.mo_energy), atol=1e-9)

    def test_mo2ao_fock(self):
        ao, mo = self.ao, self.mo
        f = basis.A(np.diag(self.scf.mo_energy), basis=(mo, mo))
        self.assertAllclose(((ao | f) | ao), self.scf.fock, atol=1e-9)
        self.assertAllclose((ao | (f | ao)), self.scf.fock, atol=1e-9)

    def test_ao2mo_dm(self):
        ao, mo = self.ao, self.mo
        d = basis.A(self.scf.dm, basis=(~ao, ~ao))#, variance=(-1, -1))
        self.assertAllclose(((mo | d) | mo), np.diag(self.scf.mo_occ))
        self.assertAllclose((mo | (d | mo)), np.diag(self.scf.mo_occ))

    def test_mo2ao_dm(self):
        ao, mo = self.ao, self.mo
        d = basis.A(np.diag(self.scf.mo_occ), basis=(mo, mo))#, variance=(-1, -1))
        self.assertAllclose(((~ao | d) | ~ao), self.scf.dm)
        self.assertAllclose((~ao | (d | ~ao)), self.scf.dm)

    @unittest.skip("Old test")
    def test_1(self):
        t2x = ccsd.t2
        mo_coeff_x = mf.mo_coeff
        occ = mf.mo_occ > 0
        vir = mf.mo_occ == 0
        mo_coeff_occ_x = mf.mo_coeff[:, mf.mo_occ > 0]
        mo_coeff_vir_x = mf.mo_coeff[:, mf.mo_occ == 0]

        # Random unitary rotation:
        u_occ = scipy.stats.ortho_group.rvs(mo_coeff_occ_x.shape[1])
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

        r_occ = mo_coeff_occ_x
        r_vir = mo_coeff_vir_x
        # ERIs are transformed without S, T is transformed with S!
        ref = np.einsum('ijab,pi,qj,ra,sb->pqrs', t2x, r_occ, r_occ, r_vir, r_vir, optimize=True)

        print(np.linalg.norm(t2x_ao.value - ref))

        t2x_mo = (occ_x, occ_x) | t2x_ao | (vir_x, vir_x)

        #t2x_mo = t2x_ao @ (vir_x, vir_x)

        print(np.linalg.norm(t2x_mo.value - t2x.value))


if __name__ == '__main__':
    unittest.main()
