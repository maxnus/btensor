from collections import namedtuple

import pytest
import numpy as np

import btensor
import btensor as basis
from btensor import Tensor, Cotensor
from helper import TestCase, rand_orth_mat
from conftest import get_subbasis_definition_random, subbasis_definition_to_matrix


scf_data = namedtuple('scf_data', ('nao', 'nmo', 'nocc', 'nvir', 'mo_coeff', 'mo_energy', 'mo_occ', 'ovlp', 'fock',
                                   'dm'))
cc_data = namedtuple('cc_data', ('mf', 'dm', 't2', 'l2'))


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
    return scf_data(nao, nmo, nocc, nvir, mo_coeff, mo_energy, mo_occ, ovlp, fock, dm)


def make_cc_data(scf):
    np.random.seed(100)
    dm = np.diag(scf.mo_occ) + np.random.uniform(-0.1, 0.1, (scf.nmo, scf.nmo))
    e, v = np.linalg.eigh(dm)
    e = np.clip(e, 0.0, 2.0)
    dm = np.dot(v*e[None], v.T)
    nvir = scf.nmo - scf.nocc
    t2 = np.random.random((scf.nocc, scf.nocc, nvir, nvir))
    l2 = np.random.random((scf.nocc, scf.nocc, nvir, nvir))
    return cc_data(scf, dm, t2, l2)


@pytest.fixture(params=[0, 0.1], ids=['Orthogonal', 'NonOrthogonal'], scope='module')
def mf(request):
    return make_scf_data(request.param)


@pytest.fixture(scope='module')
def cc(mf):
    return make_cc_data(mf)


@pytest.fixture(scope='module')
def ao(mf):
    return basis.Basis(mf.nao, metric=mf.ovlp, name='AO')


@pytest.fixture(params=[True, False], ids=['OrthKw', ''], scope='module')
def mo_orthonormal_keyword(request):
    return request.param


@pytest.fixture(scope='module')
def mo(mf, ao, mo_orthonormal_keyword):
    return basis.Basis(mf.mo_coeff, parent=ao, name='MO', orthonormal=mo_orthonormal_keyword)


def get_subbasis_definition(subtype, start, stop, size):
    if subtype == 'slice':
        return slice(start, stop)
    if subtype == 'indices':
        return list(range(start, stop))
    if subtype == 'mask':
        indices = np.arange(size)
        return (np.logical_and(indices >= start, indices < stop)).tolist()
    if subtype == 'rotation':
        return np.eye(size)[:, start:stop]
    raise ValueError(subtype)


@pytest.fixture
def mo_occ(mf, mo, subbasis_type, mo_orthonormal_keyword):
    definition = get_subbasis_definition(subbasis_type, 0, mf.nocc, mf.nmo)
    return mo.make_basis(definition, name='mo-occ', orthonormal=mo_orthonormal_keyword)


@pytest.fixture
def mo_vir(mf, mo, subbasis_type, mo_orthonormal_keyword):
    definition = get_subbasis_definition(subbasis_type, mf.nocc, mf.nmo, mf.nmo)
    return mo.make_basis(definition, name='mo-vir', orthonormal=mo_orthonormal_keyword)


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


class TestCC(TestCase):

    def test_dm_mo(self, mf, cc):
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

    def test_t2(self, mf, cc, ao, mo, mo_occ, mo_vir):
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


@pytest.fixture(params=[1, 3])
def r_occ_x(request, mo_occ, subbasis_type):
    size = request.param
    return get_subbasis_definition_random(len(mo_occ), size, subbasis_type)


@pytest.fixture(params=[1, 3])
def r_vir_x(request, mo_vir, subbasis_type):
    size = request.param
    return get_subbasis_definition_random(len(mo_vir), size, subbasis_type)


@pytest.fixture(params=[1, 3])
def r_occ_y(request, mo_occ, subbasis_type):
    size = request.param
    return get_subbasis_definition_random(len(mo_occ), size, subbasis_type)


@pytest.fixture(params=[1, 3])
def r_vir_y(request, mo_vir, subbasis_type):
    size = request.param
    return get_subbasis_definition_random(len(mo_vir), size, subbasis_type)


@pytest.fixture
def mo_occ_x(mf, mo_occ, r_occ_x, mo_orthonormal_keyword):
    return mo_occ.make_basis(r_occ_x, name='mo-occ-x', orthonormal=mo_orthonormal_keyword)


@pytest.fixture
def mo_vir_x(mf, mo_vir, r_vir_x, mo_orthonormal_keyword):
    return mo_vir.make_basis(r_vir_x, name='mo-vir-x', orthonormal=mo_orthonormal_keyword)


@pytest.fixture
def mo_occ_y(mf, mo_occ, r_occ_y, mo_orthonormal_keyword):
    return mo_occ.make_basis(r_occ_y, name='mo-occ-y', orthonormal=mo_orthonormal_keyword)


@pytest.fixture
def mo_vir_y(mf, mo_vir, r_vir_y, mo_orthonormal_keyword):
    return mo_vir.make_basis(r_vir_y, name='mo-vir-y', orthonormal=mo_orthonormal_keyword)


@pytest.fixture()
def t2x(cc, r_occ_x, r_vir_x):
    r_occ_x = subbasis_definition_to_matrix(r_occ_x, cc.mf.nocc)
    r_vir_x = subbasis_definition_to_matrix(r_vir_x, cc.mf.nvir)
    return np.einsum('ijab,iI,jJ,aA,bB->IJAB', cc.t2, r_occ_x, r_occ_x, r_vir_x, r_vir_x)


@pytest.fixture()
def l2x(cc, r_occ_x, r_vir_x):
    r_occ_x = subbasis_definition_to_matrix(r_occ_x, cc.mf.nocc)
    r_vir_x = subbasis_definition_to_matrix(r_vir_x, cc.mf.nvir)
    return np.einsum('ijab,iI,jJ,aA,bB->IJAB', cc.l2, r_occ_x, r_occ_x, r_vir_x, r_vir_x)


@pytest.fixture()
def t2y(cc, r_occ_y, r_vir_y):
    r_occ_y = subbasis_definition_to_matrix(r_occ_y, cc.mf.nocc)
    r_vir_y = subbasis_definition_to_matrix(r_vir_y, cc.mf.nvir)
    return np.einsum('ijab,iI,jJ,aA,bB->IJAB', cc.t2, r_occ_y, r_occ_y, r_vir_y, r_vir_y)


@pytest.fixture()
def l2y(cc, r_occ_y, r_vir_y):
    r_occ_y = subbasis_definition_to_matrix(r_occ_y, cc.mf.nocc)
    r_vir_y = subbasis_definition_to_matrix(r_vir_y, cc.mf.nvir)
    return np.einsum('ijab,iI,jJ,aA,bB->IJAB', cc.l2, r_occ_y, r_occ_y, r_vir_y, r_vir_y)


@pytest.fixture()
def tensor_t2x(t2x, mo_occ_x, mo_vir_x):
    return Tensor(t2x, basis=(mo_occ_x, mo_occ_x, mo_vir_x, mo_vir_x))


@pytest.fixture()
def tensor_l2y(t2y, mo_occ_y, mo_vir_y):
    return Tensor(t2y, basis=(mo_occ_y, mo_occ_y, mo_vir_y, mo_vir_y))


class TestCluster(TestCase):

    @pytest.mark.skip("WIP")
    def test_t2_l2_contraction(self, cc, r_occ_x, r_vir_x, r_occ_y, r_vir_y, t2x, l2y, tensor_t2x, tensor_l2y,
                               mo, mo_occ, mo_occ_x):
        r_occ_x = subbasis_definition_to_matrix(r_occ_x, cc.mf.nocc)
        r_vir_x = subbasis_definition_to_matrix(r_vir_x, cc.mf.nvir)
        r_occ_y = subbasis_definition_to_matrix(r_occ_y, cc.mf.nocc)
        r_vir_y = subbasis_definition_to_matrix(r_vir_y, cc.mf.nvir)
        print(mo.metric)
        print(mo.metric.to_array())
        print(mo_occ.metric)
        print(mo_occ.metric.to_array())
        print(mo_occ_x.metric.to_array())
        s_occ = np.dot(r_occ_x.T, r_occ_y)
        s_vir = np.dot(r_vir_x.T, r_vir_y)
        expected = np.einsum('ijab,KJAB,jJ,aA,bB->iK', t2x, l2y, s_occ, s_vir, s_vir)
        result = btensor.einsum('ijab,kjab->ik', tensor_t2x, tensor_l2y)
        print(result.basis)
        assert result.shape == expected.shape
        self.assert_allclose(result.to_numpy(), expected)
