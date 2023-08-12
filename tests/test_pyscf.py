#     Copyright 2023 Max Nusspickel
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from collections import namedtuple
import functools

import pytest
import numpy as np

import btensor
import btensor as basis
from btensor import Tensor, Cotensor
from helper import TestCase, rand_orth_mat
from conftest import get_random_subbasis_definition, subbasis_definition_to_matrix

scf_data = namedtuple('scf_data', ('nao', 'nmo', 'nocc', 'nvir', 'mo_coeff', 'mo_energy', 'mo_occ', 'ovlp', 'fock',
                                   'dm'))
cc_data = namedtuple('cc_data', ('mf', 'dm', 't1', 't2', 'l1', 'l2'))


def make_scf_data(nonorth):
    np.random.seed(0)
    nmo = nao = 30
    nocc = 10
    nvir = nmo - nocc
    mo_coeff = rand_orth_mat(nao) + nonorth * np.random.random((nao, nmo))
    mo_energy = np.random.uniform(-10.0, 10.0, nmo)
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
    t1 = np.random.random((scf.nocc, nvir))
    l1 = np.random.random((scf.nocc, nvir))
    t2 = np.random.random((scf.nocc, scf.nocc, nvir, nvir))
    l2 = np.random.random((scf.nocc, scf.nocc, nvir, nvir))
    return cc_data(scf, dm, t1=t1, t2=t2, l1=l1, l2=l2)


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


@pytest.fixture(scope='module')
def mo_occ(mf, mo, subbasis_type, mo_orthonormal_keyword):
    definition = get_subbasis_definition(subbasis_type, 0, mf.nocc, mf.nmo)
    return mo.make_subbasis(definition, name='mo-occ', orthonormal=mo_orthonormal_keyword)


@pytest.fixture(scope='module')
def mo_vir(mf, mo, subbasis_type, mo_orthonormal_keyword):
    definition = get_subbasis_definition(subbasis_type, mf.nocc, mf.nmo, mf.nmo)
    return mo.make_subbasis(definition, name='mo-vir', orthonormal=mo_orthonormal_keyword)


@pytest.fixture(params=['ao', 'ao_from_mo'], scope='module')
def ao2(request, mf, ao, mo):
    if request.param == 'ao':
        return ao
    r = np.dot(mf.mo_coeff.T, mf.ovlp)
    return basis.Basis(r, parent=mo)


@pytest.fixture(params=['left', 'right'], scope='module')
def double_or(request):
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

    def test_ao_mo_projector(self, mf, ao, mo):
        i = np.identity(mf.nao)
        self.assert_allclose(basis.dot(ao.get_transformation_to(mo), mo.get_transformation_to(ao)), i)

    def test_mo_ao_projector(self, mf, ao, mo):
        i = np.identity(mf.nao)
        self.assert_allclose(basis.dot(mo.get_transformation_to(ao), ao.get_transformation_to(mo)), i)

    def test_ao2mo_ovlp(self, mf, ao, mo, double_or):
        s = Cotensor(mf.ovlp, basis=(ao, ao))
        self.assert_allclose(double_or(mo, s, mo), np.identity(mf.nao))

    def test_mo2ao_ovlp(self, mf, ao, mo, double_or):
        s = Cotensor(np.identity(mf.nao), basis=(mo, mo))
        self.assert_allclose(double_or(ao, s, ao), mf.ovlp)

    def test_ao2mo_fock(self, mf, ao, mo, double_or):
        f = Cotensor(mf.fock, basis=(ao, ao))
        self.assert_allclose(double_or(mo, f, mo), np.diag(mf.mo_energy), atol=1e-9)

    def test_mo2ao_fock(self, mf, ao, mo, double_or):
        f = Cotensor(np.diag(mf.mo_energy), basis=(mo, mo))
        self.assert_allclose(double_or(ao, f, ao), mf.fock, atol=1e-9)

    def test_ao2mo_dm(self, mf, ao, mo, double_or):
        d = Tensor(mf.dm, basis=(ao, ao))
        self.assert_allclose(double_or(mo, d, mo), np.diag(mf.mo_occ))

    def test_mo2ao_dm(self, mf, ao, mo, double_or):
        d = basis.Tensor(np.diag(mf.mo_occ), basis=(mo, mo))
        self.assert_allclose(double_or(ao, d, ao), mf.dm)


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

    @pytest.fixture(scope='class')
    def t2s(self, cc, mo_occ, mo_vir):
        return basis.Tensor(cc.t2, basis=(mo_occ, mo_occ, mo_vir, mo_vir))

    @pytest.fixture(scope='class')
    def t2b(self, t2s, mo):
        return t2s[mo, mo, mo, mo]

    def test_addition(self, t2s, t2b):
        self.assert_allclose(t2s + t2b, 2 * t2b)
        self.assert_allclose(t2b + t2s, 2 * t2b)

    def test_subtraction(self, t2s, t2b):
        self.assert_allclose(t2s - t2b, 0)
        self.assert_allclose(t2b - t2s, 0)

    def test_multiplication(self, t2s, t2b, mo):
        self.assert_allclose((t2s*2).cob[mo, mo, mo, mo], t2b*2)
        self.assert_allclose((2*t2s).cob[mo, mo, mo, mo], 2*t2b)

    def test_division(self, t2s, t2b, mo):
        self.assert_allclose((t2s/2).cob[mo, mo, mo, mo], t2b/2)

    def test_trace(self, t2s, t2b, mo):
        self.assert_allclose(t2s.trace().trace(), (t2s | (mo, mo)).trace().trace())
        self.assert_allclose(t2s.trace().trace(), ((mo, mo) | t2s).trace().trace())
        self.assert_allclose(t2s.trace().trace(), t2b.trace().trace())

    def test_project(self, t2s, t2b, mo_occ, mo_vir):
        result = t2b[mo_occ, mo_occ, mo_vir, mo_vir].to_numpy()
        self.assert_allclose(result, t2s.to_numpy())

    def test_project_orthogonal_small(self, t2s, mo_occ, mo_vir):
        result = t2s[mo_vir, mo_vir, mo_occ, mo_occ]
        self.assert_allclose(result, 0)

    def test_project_orthogonal_large(self, t2b, mo_occ, mo_vir):
        result = t2b[mo_vir, mo_vir, mo_occ, mo_occ]
        self.assert_allclose(result, 0)


class TestCluster(TestCase):

    allclose_atol = 1e-10

    np_einsum = functools.partial(np.einsum, optimize=True)
    bt_einsum = functools.partial(btensor.einsum, optimize=True)

    @pytest.fixture(params=[[5, 10, 6, 9]], ids=str)
    def sizes(self, request):
        """Size of x_occ, x_vir, y_occ, y_vir"""
        return request.param

    @pytest.fixture
    def get_cluster(self, cc, mo_occ, mo_vir, subbasis_type, mo_orthonormal_keyword):
        def make_cluster(size_occ, size_vir):
            d_occ = get_random_subbasis_definition(len(mo_occ), size_occ, subbasis_type)
            d_vir = get_random_subbasis_definition(len(mo_vir), size_vir, subbasis_type)
            b_occ = mo_occ.make_subbasis(d_occ, orthonormal=mo_orthonormal_keyword)
            b_vir = mo_vir.make_subbasis(d_vir, orthonormal=mo_orthonormal_keyword)
            r_occ = subbasis_definition_to_matrix(d_occ, cc.mf.nocc)
            r_vir = subbasis_definition_to_matrix(d_vir, cc.mf.nvir)
            t1 = self.np_einsum('ia,iI,aA->IA', cc.t1, r_occ, r_vir)
            t2 = self.np_einsum('ijab,iI,jJ,aA,bB->IJAB', cc.t2, r_occ, r_occ, r_vir, r_vir)
            tensor_t1 = Tensor(t1, basis=(b_occ, b_vir))
            tensor_t2 = Tensor(t2, basis=(b_occ, b_occ, b_vir, b_vir))
            cluster = namedtuple('cluster', ('r_occ', 'r_vir', 't1', 't2', 'tensor_t1', 'tensor_t2'))
            return cluster(r_occ, r_vir, t1, t2, tensor_t1, tensor_t2)
        return make_cluster

    def test_contraction_t1_ij(self, cc, sizes, get_cluster, timings):
        size_occ_x, size_vir_x, size_occ_y, size_vir_y = sizes
        cx = get_cluster(size_occ_x, size_vir_x)
        cy = get_cluster(size_occ_y, size_vir_y)
        with timings('PySCF'):
            s_vir = np.dot(cx.r_vir.T, cy.r_vir)
            expected = self.np_einsum('ia,JB,aB->iJ', cx.t1, cy.t1, s_vir)
        with timings('BTensor'):
            result = self.bt_einsum('ia,ja->ij', cx.tensor_t1, cy.tensor_t1).to_numpy()
        self.assert_allclose(result, expected)

    def test_contraction_t1_ab(self, cc, sizes, get_cluster, timings):
        size_occ_x, size_vir_x, size_occ_y, size_vir_y = sizes
        cx = get_cluster(size_occ_x, size_vir_x)
        cy = get_cluster(size_occ_y, size_vir_y)
        with timings('PySCF'):
            s_occ = np.dot(cx.r_occ.T, cy.r_occ)
            expected = self.np_einsum('ia,JB,iJ->aB', cx.t1, cy.t1, s_occ)
        with timings('BTensor'):
            result = self.bt_einsum('ia,ib->ab', cx.tensor_t1, cy.tensor_t1).to_numpy()
        self.assert_allclose(result, expected)

    def test_contraction_t2_ik(self, cc, sizes, get_cluster, timings):
        size_occ_x, size_vir_x, size_occ_y, size_vir_y = sizes
        cx = get_cluster(size_occ_x, size_vir_x)
        cy = get_cluster(size_occ_y, size_vir_y)
        with timings('PySCF'):
            s_occ = np.dot(cx.r_occ.T, cy.r_occ)
            s_vir = np.dot(cx.r_vir.T, cy.r_vir)
            expected = self.np_einsum('ijab,KJAB,jJ,aA,bB->iK', cx.t2, cy.t2, s_occ, s_vir, s_vir)
        with timings('BTensor'):
            result = self.bt_einsum('ijab,kjab->ik', cx.tensor_t2, cy.tensor_t2).to_numpy()
        self.assert_allclose(result, expected)

    def test_contraction_t2_occ_ika(self, cc, sizes, get_cluster, timings):
        size_occ_x, size_vir_x, size_occ_y, size_vir_y = sizes
        cx = get_cluster(size_occ_x, size_vir_x)
        cy = get_cluster(size_occ_y, size_vir_y)
        with timings('PySCF'):
            s_occ = np.dot(cx.r_occ.T, cy.r_occ)
            s_vir = np.dot(cx.r_vir.T, cy.r_vir)
            expected = self.np_einsum('ijab,KJAB,jJ,bB,xa,xA->iKx', cx.t2, cy.t2, s_occ, s_vir, cx.r_vir, cy.r_vir)
        with timings('BTensor'):
            result = self.bt_einsum('ijab,kjab->ika', cx.tensor_t2, cy.tensor_t2).to_numpy()
        self.assert_allclose(result, expected)

    def test_contraction_t2_ac(self, cc, sizes, get_cluster, timings):
        size_occ_x, size_vir_x, size_occ_y, size_vir_y = sizes
        cx = get_cluster(size_occ_x, size_vir_x)
        cy = get_cluster(size_occ_y, size_vir_y)
        with timings('PySCF'):
            s_occ = np.dot(cx.r_occ.T, cy.r_occ)
            s_vir = np.dot(cx.r_vir.T, cy.r_vir)
            expected = self.np_einsum('ijab,IJCB,iI,jJ,bB->aC', cx.t2, cy.t2, s_occ, s_occ, s_vir)
        with timings('BTensor'):
            result = self.bt_einsum('ijab,ijcb->ac', cx.tensor_t2, cy.tensor_t2).to_numpy()
        self.assert_allclose(result, expected)

    #def test_contraction_t2_ac(self, cc, sizes, get_cluster, timings):
    #    size_occ_x, size_vir_x, size_occ_y, size_vir_y = sizes
    #    cx = get_cluster(size_occ_x, size_vir_x)
    #    cy = get_cluster(size_occ_y, size_vir_y)
    #    s_occ = np.dot(cx.r_occ.T, cy.r_occ)
    #    s_vir = np.dot(cx.r_vir.T, cy.r_vir)
    #    with timings('PySCF'):
    #        expected = self.np_einsum('ijab,IJCB,iI,jJ,bB->aC', cx.t2, cy.t2, s_occ, s_occ, s_vir)
    #    with timings('BTensor'):
    #        result = self.bt_einsum('ijab,ijcb->ac', cx.tensor_t2, cy.tensor_t2)
    #    self.assert_allclose(result.to_numpy(), expected)
