import pytest
import numpy as np
from collections import namedtuple
import btensor as basis
from btensor import Tensor, Cotensor
from helper import TestCase, rand_orth_mat


scf_data = namedtuple('scf_data', ('nao', 'nmo', 'nocc', 'mo_coeff', 'mo_energy', 'mo_occ', 'ovlp', 'fock', 'dm'))
cc_data = namedtuple('cc_data', ('dm', 't2'))


def make_scf_data(nonorth):
    np.random.seed(0)
    nmo = nao = 20
    nocc = 8
    nvir = nmo - nocc
    mo_coeff = rand_orth_mat(nao) + nonorth * np.random.random((nao, nmo))
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


@pytest.fixture(params=[0, 0.1], ids=['Orthogonal', 'NonOrthogonal'], scope='module')
def mf(request):
    return make_scf_data(request.param)


@pytest.fixture(scope='module')
def cc(mf):
    return make_cc_data(mf)


@pytest.fixture(scope='module')
def ao(mf):
    return basis.Basis(mf.nao, metric=mf.ovlp, name='AO')


@pytest.fixture(scope='module')
def mo(mf, ao):
    return basis.Basis(mf.mo_coeff, parent=ao, name='MO')


@pytest.fixture(params=['ao', 'ao_from_mo'], scope='module')
def ao2(request, mf, ao, mo):
    if request.param == 'ao':
        return ao
    r = np.dot(mf.mo_coeff.T, mf.ovlp)
    return basis.Basis(r, parent=mo)


@pytest.fixture(params=[np.dot, basis.dot], ids=['dot-numpy', 'dot-basis'], scope='module')
def dot_function(request):
    return request.param


@pytest.fixture(params=['left', 'right'], scope='module')
def or_association(request):

    def op(x, y, z):
        if request.param == 'left':
            return (x | y) | z
        return x | (y | z)
    return op


class TestSCF(TestCase):

    def test_cc_dm_mo(self, mf, cc):
        nocc = mf.nocc
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]
        dm = cc.dm
        dm_oo = dm[occ, occ]
        dm_ov = dm[occ, vir]
        dm_vo = dm[vir, occ]
        dm_vv = dm[vir, vir]
        mo = basis.Basis(len(mf.mo_occ))
        bo = basis.Basis(occ, parent=mo)
        bv = basis.Basis(vir, parent=mo)
        bdm_oo = basis.Tensor(dm_oo, basis=(bo, bo))
        bdm_ov = basis.Tensor(dm_ov, basis=(bo, bv))
        bdm_vo = basis.Tensor(dm_vo, basis=(bv, bo))
        bdm_vv = basis.Tensor(dm_vv, basis=(bv, bv))
        bdm = (bdm_oo + bdm_ov + bdm_vo + bdm_vv)
        self.assert_allclose(bdm._data, dm)

    def test_overlap(self, mf, ao2, mo):
        c = mf.mo_coeff
        s = mf.ovlp
        i = np.identity(mf.nao)
        self.assert_allclose(ao2.get_overlap(ao2), s)
        self.assert_allclose(mo.get_overlap(mo), i)
        self.assert_allclose(ao2.get_overlap(mo), s.dot(c))
        self.assert_allclose(mo.get_overlap(ao2), c.T.dot(s.T))

    def test_transformation_to(self, mf, ao2, mo):
        c = mf.mo_coeff
        s = mf.ovlp
        i = np.identity(mf.nao)
        # Dual-normal
        self.assert_allclose(ao2.get_transformation_to(ao2), i)
        self.assert_allclose(mo.get_transformation_to(mo), i)
        self.assert_allclose(ao2.get_transformation_to(mo), c)
        self.assert_allclose(mo.get_transformation_to(ao2), c.T.dot(s.T))

    def test_transformation_from(self, mf, ao2, mo):
        c = mf.mo_coeff
        s = mf.ovlp
        i = np.identity(mf.nao)
        # Dual-normal
        self.assert_allclose(ao2.get_transformation_from(ao2), i)
        self.assert_allclose(mo.get_transformation_from(mo), i)
        self.assert_allclose(ao2.get_transformation_from(mo), c.T.dot(s))
        self.assert_allclose(mo.get_transformation_from(ao2), c)

    def test_inverse_overlap(self, mf, ao2, mo):
        c = mf.mo_coeff
        s = mf.ovlp
        i = np.identity(mf.nao)
        self.assert_allclose(ao2.get_overlap(ao2, variance=(-1, -1)), np.linalg.inv(s))
        self.assert_allclose(mo.get_overlap(mo, variance=(-1, -1)), i)
        self.assert_allclose(ao2.get_overlap(mo, variance=(-1, -1)), c)
        self.assert_allclose(mo.get_overlap(ao2, variance=(-1, -1)), c.T)

    def test_ao_mo_projector(self, mf, ao, mo, dot_function):
        i = np.identity(mf.nao)
        self.assert_allclose(dot_function(ao.get_transformation_to(mo), mo.get_transformation_to(ao)), i)

    def test_mo_ao_projector(self, mf, ao, mo, dot_function):
        i = np.identity(mf.nao)
        self.assert_allclose(dot_function(mo.get_transformation_to(ao), ao.get_transformation_to(mo)), i)

    def test_ao2mo_ovlp(self, mf, ao, mo, or_association):
        s = Cotensor(mf.ovlp, basis=(ao, ao))
        self.assert_allclose(or_association(mo, s, mo), np.identity(mf.nao))

    def test_mo2ao_ovlp(self, mf, ao, mo, or_association):
        s = Cotensor(np.identity(mf.nao), basis=(mo, mo))
        self.assert_allclose(or_association(ao, s, ao), mf.ovlp)

    def test_ao2mo_fock(self, mf, ao, mo, or_association):
        f = Cotensor(mf.fock, basis=(ao, ao))
        self.assert_allclose(or_association(mo, f, mo), np.diag(mf.mo_energy), atol=1e-9)

    def test_mo2ao_fock(self, mf, ao, mo, or_association):
        f = Cotensor(np.diag(mf.mo_energy), basis=(mo, mo))
        self.assert_allclose(or_association(ao, f, ao), mf.fock, atol=1e-9)

    def test_ao2mo_dm(self, mf, ao, mo, or_association):
        d = Tensor(mf.dm, basis=(ao, ao))
        self.assert_allclose(or_association(mo, d, mo), np.diag(mf.mo_occ))

    def test_mo2ao_dm(self, mf, ao, mo, or_association):
        d = basis.Tensor(np.diag(mf.mo_occ), basis=(mo, mo))
        self.assert_allclose(or_association(ao, d, ao), mf.dm)

    def test_mo2mo_t2(self, mf, cc, ao, mo):
        mo_occ = basis.Basis(slice(mf.nocc), parent=mo)
        mo_vir = basis.Basis(slice(mf.nocc, mf.nmo), parent=mo)
        t2s = basis.Tensor(cc.t2, basis=(mo_occ, mo_occ, mo_vir, mo_vir))
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
        t2r = t2b.project((mo_occ, mo_occ, mo_vir, mo_vir))
        self.assert_allclose(t2s._data, t2r._data)
        # Empty array for orthogonal basis
        t2e = t2s.project((mo_vir, mo_vir, mo_occ, mo_occ))
        self.assert_allclose(t2e, 0)
        t2e = t2b.project((mo_vir, mo_vir, mo_occ, mo_occ))
        self.assert_allclose(t2e, 0)

    def test_foo(self, ao):
        l = {ao : 2}
