import numpy as np
from collections import namedtuple
import btensor as basis
from testing import TestCase, rand_orth_mat


scf_data = namedtuple('scf_data', ('nao', 'nmo', 'nocc', 'mo_coeff', 'mo_energy', 'mo_occ', 'ovlp', 'fock', 'dm'))
cc_data = namedtuple('cc_data', ('dm', 't2'))


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
    nvir = scf.nmo - scf.nocc
    t2 = np.random.random((scf.nocc, scf.nocc, nvir, nvir))
    return cc_data(dm, t2)


class TestSCF(TestCase):

    @classmethod
    def setup_class(cls):
        cls.scf = make_scf_data(20)
        cls.cc = make_cc_data(cls.scf)
        cls.ao = basis.Basis(cls.scf.nao, metric=cls.scf.ovlp, name='AO')
        cls.mo = basis.Basis(cls.scf.mo_coeff, parent=cls.ao, name='MO')

    def test_cc_dm_mo(self):
        nocc = self.scf.nocc
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]
        dm = self.cc.dm
        dm_oo = dm[occ, occ]
        dm_ov = dm[occ, vir]
        dm_vo = dm[vir, occ]
        dm_vv = dm[vir, vir]
        mo = basis.Basis(len(self.scf.mo_occ))
        #occ = np.arange(self.scf.nmo)[occ]
        #vir = np.arange(self.scf.nmo)[vir]
        bo = basis.Basis(occ, parent=mo)
        bv = basis.Basis(vir, parent=mo)
        bdm_oo = basis.Tensor(dm_oo, basis=(bo, bo))
        bdm_ov = basis.Tensor(dm_ov, basis=(bo, bv))
        bdm_vo = basis.Tensor(dm_vo, basis=(bv, bo))
        bdm_vv = basis.Tensor(dm_vv, basis=(bv, bv))
        bdm = (bdm_oo + bdm_ov + bdm_vo + bdm_vv)
        self.assert_allclose(bdm._data, dm)

    def test_ao_mo_transform(self):
        ao, mo = self.ao, self.mo
        c = self.scf.mo_coeff
        s = self.scf.ovlp
        i = np.identity(self.scf.nao)

        # Normal-normal
        self.assert_allclose((ao | ao), s)
        self.assert_allclose((mo | mo), i)
        self.assert_allclose((ao | mo), s.dot(c))
        self.assert_allclose((mo | ao), c.T.dot(s.T))
        # Dual-normal
        self.assert_allclose((-ao | ao), i)
        self.assert_allclose((-mo | mo), i)
        self.assert_allclose((-ao | mo), c)
        self.assert_allclose((-mo | ao), c.T.dot(s.T))
        # Normal-dual
        self.assert_allclose((ao | -ao), i)
        self.assert_allclose((mo | -mo), i)
        self.assert_allclose((ao | -mo), s.dot(c))
        self.assert_allclose((mo | -ao), c.T)
        # Dual-dual
        self.assert_allclose((-ao | -ao), np.linalg.inv(s))
        self.assert_allclose((-mo | -mo), i)
        self.assert_allclose((-ao | -mo), c)
        self.assert_allclose((-mo | -ao), c.T)

        # AO derived from MO
        r = np.dot(self.scf.mo_coeff.T, self.scf.ovlp)
        ao = basis.Basis(r, parent=mo)
        # Normal-normal
        self.assert_allclose((ao | ao), s)
        self.assert_allclose((mo | mo), i)
        self.assert_allclose((ao | mo), s.dot(c))
        self.assert_allclose((mo | ao), c.T.dot(s.T))
        # Dual-normal
        self.assert_allclose((-ao | ao), i)
        self.assert_allclose((-mo | mo), i)
        self.assert_allclose((-ao | mo), c)
        self.assert_allclose((-mo | ao), c.T.dot(s.T))
        # Normal-dual
        self.assert_allclose((ao | -ao), i)
        self.assert_allclose((mo | -mo), i)
        self.assert_allclose((ao | -mo), s.dot(c))
        self.assert_allclose((mo | -ao), c.T)
        # Dual-dual
        self.assert_allclose((-ao | -ao), np.linalg.inv(s))
        self.assert_allclose((-mo | -mo), i)
        self.assert_allclose((-ao | -mo), c)
        self.assert_allclose((-mo | -ao), c.T)

    def test_ao_mo_projector(self):
        ao, mo = self.ao, self.mo
        i = np.identity(self.scf.nao)
        # NumPy
        # 1
        self.assert_allclose(np.dot((-ao | mo), (mo | ao)), i)
        self.assert_allclose(np.dot((-ao | -mo), (mo | ao)), i)
        self.assert_allclose(np.dot((-ao | mo), (-mo | ao)), i)
        self.assert_allclose(np.dot((-ao | -mo), (-mo | ao)), i)
        # 2
        self.assert_allclose(np.dot((mo | -ao), (ao | mo)), i)
        self.assert_allclose(np.dot((-mo | -ao), (ao | mo)), i)
        self.assert_allclose(np.dot((mo | -ao), (ao | -mo)), i)
        self.assert_allclose(np.dot((-mo | -ao), (ao | -mo)), i)
        # 3
        self.assert_allclose(np.dot((mo | ao), (-ao | mo)), i)
        self.assert_allclose(np.dot((-mo | ao), (-ao | mo)), i)
        self.assert_allclose(np.dot((mo | ao), (-ao | -mo)), i)
        self.assert_allclose(np.dot((-mo | ao), (-ao | -mo)), i)
        # 4
        self.assert_allclose(np.dot((ao | mo), (mo | -ao)), i)
        self.assert_allclose(np.dot((ao | -mo), (mo | -ao)), i)
        self.assert_allclose(np.dot((ao | mo), (-mo | -ao)), i)
        self.assert_allclose(np.dot((ao | -mo), (-mo | -ao)), i)

        # basis
        # 1
        self.assert_allclose(basis.dot((-ao | mo), (mo | ao)), i)
        self.assert_allclose(basis.dot((-ao | -mo), (mo | ao)), i)
        self.assert_allclose(basis.dot((-ao | mo), (-mo | ao)), i)
        self.assert_allclose(basis.dot((-ao | -mo), (-mo | ao)), i)
        # 2
        self.assert_allclose(basis.dot((mo | -ao), (ao | mo)), i)
        self.assert_allclose(basis.dot((-mo | -ao), (ao | mo)), i)
        self.assert_allclose(basis.dot((mo | -ao), (ao | -mo)), i)
        self.assert_allclose(basis.dot((-mo | -ao), (ao | -mo)), i)
        # 3
        self.assert_allclose(basis.dot((mo | ao), (-ao | mo)), i)
        self.assert_allclose(basis.dot((-mo | ao), (-ao | mo)), i)
        self.assert_allclose(basis.dot((mo | ao), (-ao | -mo)), i)
        self.assert_allclose(basis.dot((-mo | ao), (-ao | -mo)), i)
        # 4
        self.assert_allclose(basis.dot((ao | mo), (mo | -ao)), i)
        self.assert_allclose(basis.dot((ao | -mo), (mo | -ao)), i)
        self.assert_allclose(basis.dot((ao | mo), (-mo | -ao)), i)
        self.assert_allclose(basis.dot((ao | -mo), (-mo | -ao)), i)

    def test_ao2mo_ovlp(self):
        ao, mo = self.ao, self.mo
        s = basis.Tensor(self.scf.ovlp, basis=(-ao, -ao))
        self.assert_allclose(((mo | s) | mo), np.identity(self.scf.nao))
        self.assert_allclose((mo | (s | mo)), np.identity(self.scf.nao))
        s = basis.Cotensor(self.scf.ovlp, basis=(ao, ao))
        self.assert_allclose(((mo | s) | mo), np.identity(self.scf.nao))
        self.assert_allclose((mo | (s | mo)), np.identity(self.scf.nao))

    def test_mo2ao_ovlp(self):
        ao, mo = self.ao, self.mo
        s = basis.Tensor(np.identity(self.scf.nao), basis=(mo, mo))
        self.assert_allclose(((-ao | s) | -ao), self.scf.ovlp)
        self.assert_allclose((-ao | (s | -ao)), self.scf.ovlp)
        s = basis.Cotensor(np.identity(self.scf.nao), basis=(mo, mo))
        self.assert_allclose(((ao | s) | ao), self.scf.ovlp)
        self.assert_allclose((ao | (s | ao)), self.scf.ovlp)

    def test_ao2mo_fock(self):
        ao, mo = self.ao, self.mo
        f = basis.Tensor(self.scf.fock, basis=(-ao, -ao))
        self.assert_allclose(((mo | f) | mo), np.diag(self.scf.mo_energy), atol=1e-9)
        self.assert_allclose((mo | (f | mo)), np.diag(self.scf.mo_energy), atol=1e-9)
        f = basis.Cotensor(self.scf.fock, basis=(ao, ao))
        self.assert_allclose(((mo | f) | mo), np.diag(self.scf.mo_energy), atol=1e-9)
        self.assert_allclose((mo | (f | mo)), np.diag(self.scf.mo_energy), atol=1e-9)

    def test_mo2ao_fock(self):
        ao, mo = self.ao, self.mo
        f = basis.Tensor(np.diag(self.scf.mo_energy), basis=(mo, mo))
        self.assert_allclose(((-ao | f) | -ao), self.scf.fock, atol=1e-9)
        self.assert_allclose((-ao | (f | -ao)), self.scf.fock, atol=1e-9)
        f = basis.Cotensor(np.diag(self.scf.mo_energy), basis=(mo, mo))
        self.assert_allclose(((ao | f) | ao), self.scf.fock, atol=1e-9)
        self.assert_allclose((ao | (f | ao)), self.scf.fock, atol=1e-9)

    def test_ao2mo_dm(self):
        ao, mo = self.ao, self.mo
        d = basis.Tensor(self.scf.dm, basis=(ao, ao))
        self.assert_allclose(((mo | d) | mo), np.diag(self.scf.mo_occ))
        self.assert_allclose((mo | (d | mo)), np.diag(self.scf.mo_occ))
        d = basis.Cotensor(self.scf.dm, basis=(-ao, -ao))
        self.assert_allclose(((mo | d) | mo), np.diag(self.scf.mo_occ))
        self.assert_allclose((mo | (d | mo)), np.diag(self.scf.mo_occ))

    def test_mo2ao_dm(self):
        ao, mo = self.ao, self.mo
        d = basis.Tensor(np.diag(self.scf.mo_occ), basis=(mo, mo))
        self.assert_allclose(((ao | d) | ao), self.scf.dm)
        self.assert_allclose((ao | (d | ao)), self.scf.dm)
        d = basis.Cotensor(np.diag(self.scf.mo_occ), basis=(mo, mo))
        self.assert_allclose(((-ao | d) | -ao), self.scf.dm)
        self.assert_allclose((-ao | (d | -ao)), self.scf.dm)

    def test_mo2mo_t2(self):
        ao, mo = self.ao, self.mo
        mo_occ = basis.Basis(slice(self.scf.nocc), parent=mo)
        mo_vir = basis.Basis(slice(self.scf.nocc, self.scf.nmo), parent=mo)
        t2s = basis.Tensor(self.cc.t2, basis=(mo_occ, mo_occ, mo_vir, mo_vir))
        t2b = (mo, mo) | t2s | (mo, mo)
        # Add
        self.assert_allclose(t2s + t2b, 2 * t2b)
        self.assert_allclose(t2b + t2s, 2 * t2b)
        # Subtract
        self.assert_allclose(t2s - t2b, 0)
        self.assert_allclose(t2b - t2s, 0)
        # Multiply
        self.assert_allclose(t2s * t2b, t2b * t2b)
        self.assert_allclose(t2b * t2s, t2b * t2b)
        # Divide
        self.assert_allclose(t2s / (abs(t2b) + 1), t2b / (abs(t2b) + 1))
        # Floor Divide
        self.assert_allclose(t2s // (abs(t2b) + 1), t2b // (abs(t2b) + 1))
        # Modulus
        self.assert_allclose(t2s % t2b, t2b % t2b)
        self.assert_allclose(t2b % t2s, t2b % t2b)
        # Power
        self.assert_allclose(t2s ** t2b, t2b ** t2b)
        self.assert_allclose(t2b ** t2s, t2b ** t2b)
        # Preserved trace
        self.assert_allclose(t2s.trace().trace(), (t2s | (mo, mo)).trace().trace())
        self.assert_allclose(t2s.trace().trace(), ((mo, mo) | t2s).trace().trace())
        self.assert_allclose(t2s.trace().trace(), t2b.trace().trace())
        # Restore
        t2r = t2b.proj((mo_occ, mo_occ, mo_vir, mo_vir))
        self.assert_allclose(t2s._data, t2r._data)
        # Empty array for orthogonal basis
        t2e = t2s.proj((mo_vir, mo_vir, mo_occ, mo_occ))
        self.assert_allclose(t2e, 0)
        t2e = t2b.proj((mo_vir, mo_vir, mo_occ, mo_occ))
        self.assert_allclose(t2e, 0)
