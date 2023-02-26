import unittest
import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import scipy
import scipy.stats

#from basis_array import Basis, RootBasis, Array, einsum
import basis_array as basis


def rand_orth_mat(n, ncol=None):
    m = scipy.stats.ortho_group.rvs(n)
    #return m
    if ncol is not None:
        m = m[:,:ncol]
    return m


class TestCase(unittest.TestCase):

    allclose_atol = 1e-8
    allclose_rtol = 1e-7

    def assertAllclose(self, actual, desired, rtol=allclose_atol, atol=allclose_rtol, **kwargs):
        # Compare multiple pairs of arrays:
        if isinstance(actual, (tuple, list)):
            for i in range(len(actual)):
                self.assertAllclose(actual[i], desired[i], rtol=rtol, atol=atol, **kwargs)
            return
        # Compare single pair of arrays:
        try:
            np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, **kwargs)
        except AssertionError as e:
            # Add higher precision output:
            message = e.args[0]
            args = e.args[1:]
            message += '\nHigh precision:\n x: %r\n y: %r' % (actual, desired)
            e.args = (message, *args)
            raise


class ArithmetricTestsSameBasis(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        b1 = basis.B(n1)
        b2 = basis.B(n2)
        cls.d1 = np.random.rand(n1, n2)
        cls.d2 = np.random.rand(n1, n2)
        cls.a1 = basis.Array(cls.d1, basis=(b1, b2))
        cls.a2 = basis.Array(cls.d2, basis=(b1, b2))

    def test_negation(self):
        self.assertAllclose((-self.a1).value, -self.d1)

    def test_absolute(self):
        self.assertAllclose(abs(self.a1).value, abs(self.d1))

    def test_addition(self):
        self.assertAllclose((self.a1 + self.a2).value, self.d1+self.d2)

    def test_subtraction(self):
        self.assertAllclose((self.a1 - self.a2).value, self.d1-self.d2)

    def test_multiplication(self):
        self.assertAllclose((self.a1 * self.a2).value, self.d1 * self.d2)

    def test_division(self):
        self.assertAllclose((self.a1 / self.a2).value, self.d1 / self.d2)

    def test_floordivision(self):
        self.assertAllclose((self.a1 // self.a2).value, self.d1 // self.d2)

    def test_getitem_int(self):
        self.assertAllclose(self.a1[0].value, self.d1[0])
        self.assertAllclose(self.a1[-1].value, self.d1[-1])

    def test_getitem_slice(self):
        self.assertAllclose(self.a1[:2].value, self.d1[:2])
        self.assertAllclose(self.a1[::2].value, self.d1[::2])
        self.assertAllclose(self.a1[:-1].value, self.d1[:-1])

    def test_getitem_elipsis(self):
        self.assertAllclose(self.a1[...].value, self.d1[...])

    def test_getitem_tuple(self):
        self.assertAllclose(self.a1[0,2].value, self.d1[0,2])
        self.assertAllclose(self.a1[-1,2].value, self.d1[-1,2])
        self.assertAllclose(self.a1[:2,3:].value, self.d1[:2,3:])
        self.assertAllclose(self.a1[::2,::-1].value, self.d1[::2,::-1])
        self.assertAllclose(self.a1[0,:2].value, self.d1[0,:2])
        self.assertAllclose(self.a1[::-1,-1].value, self.d1[::-1,-1])
        self.assertAllclose(self.a1[...,0], self.d1[...,0])
        self.assertAllclose(self.a1[...,:2], self.d1[...,:2])
        self.assertAllclose(self.a1[::-1,...], self.d1[::-1,...])

    #def test_getitem_boolean(self):
    #    mask = [True, True, False, False, True]
    #    self.assertAllclose(self.a1[mask].value, self.d1[mask])


class ArithmetricTestsDifferentBasis(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        n11, n12 = 3, 4
        n21, n22 = 6, 2
        b1 = basis.B(n1)
        b2 = basis.B(n2)
        cls.c11 = rand_orth_mat(n1, n11)
        cls.c12 = rand_orth_mat(n1, n12)
        cls.c21 = rand_orth_mat(n2, n21)
        cls.c22 = rand_orth_mat(n2, n22)
        b11 = basis.B(b1, cls.c11)
        b12 = basis.B(b1, cls.c12)
        b21 = basis.B(b2, cls.c21)
        b22 = basis.B(b2, cls.c22)
        cls.d1 = np.random.rand(n11, n21)
        cls.d2 = np.random.rand(n12, n22)
        cls.a1 = basis.Array(cls.d1, basis=(b11, b21))
        cls.a2 = basis.Array(cls.d2, basis=(b12, b22))
        # To check
        cls.t1 = np.linalg.multi_dot((cls.c11, cls.d1, cls.c21.T))
        cls.t2 = np.linalg.multi_dot((cls.c12, cls.d2, cls.c22.T))

    def test_addition(self):
        self.assertAllclose((self.a1 + self.a2).value, self.t1 + self.t2)

    def test_subtraction(self):
        self.assertAllclose((self.a1 - self.a2).value, self.t1 - self.t2)

    def test_multiplication(self):
        self.assertAllclose((self.a1 * self.a2).value, self.t1 * self.t2)

    def test_division(self):
        self.assertAllclose((self.a1 / self.a2).value, self.t1 / self.t2)


class SubbasisTests(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        n1, n2 = 5, 6
        m1, m2 = 3, 4
        b1 = basis.B(n1)
        b2 = basis.B(n2)
        cls.c1 = rand_orth_mat(n1, m1)
        cls.c2 = rand_orth_mat(n2, m2)
        sb1 = basis.B(b1, cls.c1)
        sb2 = basis.B(b2, cls.c2)
        cls.d1 = np.random.rand(n1, n2)
        cls.d2 = np.linalg.multi_dot((cls.c1.T, cls.d1, cls.c2))
        cls.a1 = basis.Array(cls.d1, basis=(b1, b2))
        cls.a2 = basis.Array(cls.d2, basis=(sb1, sb2))

    def test_basis_setter(self):
        a1a = self.a1.as_basis(self.a2.basis)
        a1b = self.a1.copy()
        a1b.basis = self.a2.basis
        self.assertAllclose(a1a, a1b)

    def test_subspace(self):
        a2 = self.a1.as_basis(self.a2.basis)
        self.assertAllclose(self.a2, a2)


class PySCF_Tests(TestCase):


    @classmethod
    def setUpClass(cls):
        return
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = 'Li 0 0 0 ; Li 0 0 1.5'
        #cls.mol.basis = 'cc-pVDZ'
        cls.mol.basis = 'sto-3g'
        cls.mol.build()
        cls.mf = pyscf.scf.HF(cls.mol)
        cls.mf.kernel()
        cls.cc = pyscf.cc.CCSD(cls.mf)
        cls.cc.kernel()

    def test_dm(self):
        return
        mf = self.mf
        nocc = sum(mf.mo_occ>0)
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]
        c_occ = mf.mo_coeff[:,occ]
        c_vir = mf.mo_coeff[:,vir]

        dm = self.cc.make_rdm1()
        dm_oo = dm[occ,occ]
        dm_ov = dm[occ,vir]
        dm_vo = dm[vir,occ]
        dm_vv = dm[vir,vir]

        mo = basis.B(len(mf.mo_occ))
        bo = basis.B(mo, occ)
        bv = basis.B(mo, vir)
        bdm_oo = basis.A(dm_oo, basis=(bo, bo))
        bdm_ov = basis.A(dm_ov, basis=(bo, bv))
        bdm_vo = basis.A(dm_vo, basis=(bv, bo))
        bdm_vv = basis.A(dm_vv, basis=(bv, bv))
        bdm = (bdm_oo + bdm_ov + bdm_vo + bdm_vv)

        self.assertAllclose(bdm.value, dm)

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
