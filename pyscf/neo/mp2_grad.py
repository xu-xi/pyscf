#!/usr/bin/env python
# Analytic gradients for CNEO-MP2 and CNEO-MP2(ee)

import numpy
from functools import reduce
from pyscf import lib, ao2mo
from pyscf.ao2mo import _ao2mo
from . import hessian, grad as neo_grad
from pyscf.lib import logger
from pyscf.grad.mp2 import has_frozen_orbitals, _shell_prange
from .mp2 import _ep_ao_eri, _ep_ovov_from_ao
from .mp2_grad_slow import (ee_corr_grad, ep_corr_grad,
                            _ao_eri_deriv_ep_ovov_cross,
                            _fill_canonical_mo_response)


def _mo_density(mo_coeff, dm1mo):
    return reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))


def _ao2mo_block(mol, coeffs):
    shape = tuple(c.shape[1] for c in coeffs)
    return ao2mo.general(mol, coeffs, compact=False).reshape(shape)


def _block_size(max_mb, *dims):
    bytes_per_item = numpy.dtype(float).itemsize
    denom = bytes_per_item
    for n in dims:
        denom *= max(int(n), 1)
    return max(1, int(max_mb * 1e6 // denom))


class _EEMOBlocks:
    def __init__(self, mol, mo_coeff, nocc, max_block_mb=256):
        self.mol = mol
        self.mo_coeff = mo_coeff
        self.nocc = nocc
        self.nmo = mo_coeff.shape[1]
        self.nvir = self.nmo - nocc
        self.co = mo_coeff[:, :nocc]
        self.cv = mo_coeff[:, nocc:]
        self.max_block_mb = max_block_mb
        self._ovov = None
        self._ooov = None

    @property
    def ovov(self):
        if self._ovov is None:
            self._ovov = _ao2mo_block(self.mol,
                                      (self.co, self.cv, self.co, self.cv))
        return self._ovov

    @property
    def ooov(self):
        if self._ooov is None:
            self._ooov = _ao2mo_block(self.mol,
                                      (self.co, self.co, self.co, self.cv))
        return self._ooov

    def vvov_chunk_size(self):
        return _block_size(self.max_block_mb, self.nvir,
                           self.nocc, self.nvir)

    def ovvv_chunk_size(self):
        return _block_size(self.max_block_mb, self.nocc,
                           self.nvir, self.nvir)

    def vvov_chunk(self, p0, p1):
        return _ao2mo_block(self.mol, (self.cv[:, p0:p1], self.cv,
                                      self.co, self.cv))

    def ovvv_chunk(self, p0, p1):
        return _ao2mo_block(self.mol, (self.co, self.cv,
                                      self.cv[:, p0:p1], self.cv))


class _EPMOBlocks:
    def __init__(self, mol_e, mol_p, ce, cp, occ_e, vir_e, occ_p, vir_p,
                 charge_product):
        self.mol_e = mol_e
        self.mol_p = mol_p
        self.ce = ce
        self.cp = cp
        self.occ_e = occ_e
        self.vir_e = vir_e
        self.occ_p = occ_p
        self.vir_p = vir_p
        self.charge_product = charge_product
        self.coe = ce[:, occ_e]
        self.cve = ce[:, vir_e]
        self.cop = cp[:, occ_p]
        self.cvp = cp[:, vir_p]
        self.nocc_e = self.coe.shape[1]
        self.nvir_e = self.cve.shape[1]
        self.nocc_p = self.cop.shape[1]
        self.nvir_p = self.cvp.shape[1]
        self._eri_ao = None
        self._paIA = None
        self._ipIA = None
        self._iaPA = None
        self._iaIP = None

    @property
    def eri_ao(self):
        if self._eri_ao is None:
            self._eri_ao = _ep_ao_eri(self.mol_e, self.mol_p)
        return self._eri_ao

    def _transform(self, coeffs):
        return (_ep_ovov_from_ao(self.eri_ao, *coeffs) *
                self.charge_product)

    @property
    def paIA(self):
        if self._paIA is None:
            self._paIA = self._transform((self.ce, self.cve,
                                          self.cop, self.cvp))
        return self._paIA

    @property
    def ipIA(self):
        if self._ipIA is None:
            self._ipIA = self._transform((self.coe, self.ce,
                                          self.cop, self.cvp))
        return self._ipIA

    @property
    def iaPA(self):
        if self._iaPA is None:
            self._iaPA = self._transform((self.coe, self.cve,
                                          self.cp, self.cvp))
        return self._iaPA

    @property
    def iaIP(self):
        if self._iaIP is None:
            self._iaIP = self._transform((self.coe, self.cve,
                                          self.cop, self.cp))
        return self._iaIP

    def drop_ao_cache(self):
        self._eri_ao = None
        return self


def _ee_deriv_contract(mol, co, cv, t2, atmlst, max_memory=None):
    '''Contract ee derivative ERIs with MP2 amplitudes in AO shell blocks.'''
    nao, nocc = co.shape
    nvir = cv.shape[1]
    t2 = numpy.asarray(t2)

    # Same partial two-particle density transformation as conventional MP2
    # gradients.  It avoids forming derivative MO integrals such as
    # d(ov|ov), whose AO->MO intermediates are too large for these clusters.
    part_dm2 = _ao2mo.nr_e2(t2.reshape(nocc**2, nvir**2),
                            numpy.asarray(cv.T, order='F'),
                            (0, nao, 0, nao), 's1', 's1')
    part_dm2 = part_dm2.reshape(nocc, nocc, nao, nao)
    part_dm2 = (part_dm2.transpose(0, 2, 3, 1) * 4 -
                part_dm2.transpose(0, 3, 2, 1) * 2)

    if max_memory is None:
        max_memory = max(200, mol.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))

    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    offset = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst), 3), dtype=numpy.result_type(t2, co))

    for k, ia in enumerate(atmlst):
        sh0, sh1, p0, p1 = offset[ia]
        ip1 = p0
        for b0, b1, nf in _shell_prange(mol, sh0, sh1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = lib.einsum('pi,iqrj->pqrj',
                                co[ip0:ip1], part_dm2)
            dm2buf += lib.einsum('qi,iprj->pqrj',
                                 co, part_dm2[:, ip0:ip1])
            dm2buf = lib.einsum('pqrj,sj->pqrs', dm2buf, co)
            dm2buf = dm2buf + dm2buf.transpose(0, 1, 3, 2)
            dm2buf = lib.pack_tril(
                dm2buf.reshape(-1, nao, nao)).reshape(nf, nao, -1)
            dm2buf[:, :, diagidx] *= .5

            shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3, nf, nao, -1)
            de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2
    return de


def _eri_ovov_rotation_deriv_blocked(eri, u):
    nocc = eri.nocc
    nvir = eri.nvir
    uo = u[:, :nocc]
    uv = u[:, nocc:]
    ovov = eri.ovov
    ooov = eri.ooov

    d = numpy.einsum('pi,pajb->iajb', uo[:nocc], ovov)
    d += numpy.einsum('pa,ipjb->iajb', uv[:nocc], ooov)
    d += numpy.einsum('pj,iapb->iajb', uo[:nocc], ovov)
    d += numpy.einsum('pb,jpia->iajb', uv[:nocc], ooov)

    d += numpy.einsum('pa,ipjb->iajb', uv[nocc:], ovov)
    d += numpy.einsum('pb,iajp->iajb', uv[nocc:], ovov)

    for p0 in range(0, nvir, eri.vvov_chunk_size()):
        p1 = min(p0 + eri.vvov_chunk_size(), nvir)
        d += numpy.einsum('pi,pajb->iajb', uo[nocc+p0:nocc+p1],
                          eri.vvov_chunk(p0, p1))

    for p0 in range(0, nvir, eri.ovvv_chunk_size()):
        p1 = min(p0 + eri.ovvv_chunk_size(), nvir)
        d += numpy.einsum('pj,iapb->iajb', uo[nocc+p0:nocc+p1],
                          eri.ovvv_chunk(p0, p1))

    return d.transpose(0, 2, 1, 3)


def _eri_ovov_rotation_contract(eri, u, t2_bar):
    return _ee_rotation_terms_contract(_ee_rotation_terms(eri, t2_bar), u)


def _ee_rotation_terms(eri, t2_bar):
    nocc = eri.nocc
    nvir = eri.nvir
    ovov = eri.ovov
    ooov = eri.ooov
    w = 2.0 * t2_bar.transpose(0, 2, 1, 3)

    uo_occ = numpy.einsum('iajb,pajb->pi', w, ovov)
    uo_occ += numpy.einsum('iajb,iapb->pj', w, ovov)
    uv_occ = numpy.einsum('iajb,ipjb->pa', w, ooov)
    uv_occ += numpy.einsum('iajb,jpia->pb', w, ooov)
    uv_vir = numpy.einsum('iajb,ipjb->pa', w, ovov)
    uv_vir += numpy.einsum('iajb,iajp->pb', w, ovov)
    uo_vir = numpy.zeros((nvir, nocc), dtype=numpy.result_type(w, ovov))

    for p0 in range(0, nvir, eri.vvov_chunk_size()):
        p1 = min(p0 + eri.vvov_chunk_size(), nvir)
        uo_vir[p0:p1] += numpy.einsum('iajb,pajb->pi', w,
                                      eri.vvov_chunk(p0, p1))

    for p0 in range(0, nvir, eri.ovvv_chunk_size()):
        p1 = min(p0 + eri.ovvv_chunk_size(), nvir)
        uo_vir[p0:p1] += numpy.einsum('iajb,iapb->pj', w,
                                      eri.ovvv_chunk(p0, p1))

    return nocc, uo_occ, uo_vir, uv_occ, uv_vir


def _ee_rotation_terms_contract(terms, u):
    nocc, uo_occ, uo_vir, uv_occ, uv_vir = terms
    uo = u[:, :nocc]
    uv = u[:, nocc:]
    return (numpy.einsum('pi,pi->', uo[:nocc], uo_occ) +
            numpy.einsum('pi,pi->', uo[nocc:], uo_vir) +
            numpy.einsum('pa,pa->', uv[:nocc], uv_occ) +
            numpy.einsum('pa,pa->', uv[nocc:], uv_vir))


def _ee_denom_deriv_weights(t2, t2_bar):
    w_occ = (numpy.einsum('ijab,ijab->i', t2, t2_bar) +
             numpy.einsum('ijab,ijab->j', t2, t2_bar))
    w_vir = (numpy.einsum('ijab,ijab->a', t2, t2_bar) +
             numpy.einsum('ijab,ijab->b', t2, t2_bar))
    return w_occ, w_vir


def _ee_denom_deriv_contract(eps1, nocc, weights):
    w_occ, w_vir = weights
    return (numpy.dot(eps1[:nocc], w_occ) -
            numpy.dot(eps1[nocc:], w_vir))


def _ep_denom_deriv_weights(t2):
    w_occ_e = numpy.einsum('iaIA,iaIA->i', t2, t2)
    w_vir_e = numpy.einsum('iaIA,iaIA->a', t2, t2)
    w_occ_p = numpy.einsum('iaIA,iaIA->I', t2, t2)
    w_vir_p = numpy.einsum('iaIA,iaIA->A', t2, t2)
    return w_occ_e, w_vir_e, w_occ_p, w_vir_p


def _ep_denom_deriv_contract(eps1_e, eps1_p, nocc_e, nocc_p, weights):
    w_occ_e, w_vir_e, w_occ_p, w_vir_p = weights
    return (numpy.dot(eps1_e[:nocc_e], w_occ_e) -
            numpy.dot(eps1_e[nocc_e:], w_vir_e) +
            numpy.dot(eps1_p[:nocc_p], w_occ_p) -
            numpy.dot(eps1_p[nocc_p:], w_vir_p))


def _ep_ovov_rotation_contract(eri, ue, up, t2):
    return _ep_rotation_terms_contract(_ep_rotation_terms(eri, t2), ue, up)


def _ep_rotation_terms(eri, t2):
    w = 4.0 * t2
    ue_occ = numpy.einsum('iaIA,paIA->pi', w, eri.paIA)
    ue_vir = numpy.einsum('iaIA,ipIA->pa', w, eri.ipIA)
    up_occ = numpy.einsum('iaIA,iaPA->PI', w, eri.iaPA)
    up_vir = numpy.einsum('iaIA,iaIP->PA', w, eri.iaIP)
    return eri.nocc_e, eri.nocc_p, ue_occ, ue_vir, up_occ, up_vir


def _ep_rotation_terms_contract(terms, ue, up):
    nocc_e, nocc_p, ue_occ, ue_vir, up_occ, up_vir = terms
    return (numpy.einsum('pi,pi->', ue[:, :nocc_e], ue_occ) +
            numpy.einsum('pa,pa->', ue[:, nocc_e:], ue_vir) +
            numpy.einsum('PI,PI->', up[:, :nocc_p], up_occ) +
            numpy.einsum('PA,PA->', up[:, nocc_p:], up_vir))


def _s1mo_for_atom(comp, ia):
    s1a = -comp.mol.intor('int1e_ipovlp', comp=3)
    nao = comp.mol.nao_nr()
    p0, p1 = comp.mol.aoslice_by_atom()[ia][2:]
    s1ao = numpy.zeros((3, nao, nao), dtype=s1a.dtype)
    s1ao[:, p0:p1] += s1a[:, p0:p1]
    s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
    return numpy.asarray([reduce(numpy.dot, (comp.mo_coeff.T, s1ao[x],
                                             comp.mo_coeff))
                          for x in range(3)])


def _ep_intermediates(mf, t2_ep):
    '''Build ep density, bare Z-vector RHS, and signed ep MO integrals.'''
    comp_e = mf.components['e']
    ce = comp_e.mo_coeff
    occ_e = comp_e.mo_occ > 0
    vir_e = comp_e.mo_occ == 0
    nocc_e = numpy.count_nonzero(occ_e)
    nvir_e = numpy.count_nonzero(vir_e)

    dm1mo_e = numpy.zeros((ce.shape[1], ce.shape[1]))
    lvo_e = numpy.zeros((nvir_e, nocc_e))
    dm1mo_p = {}
    lvo_p = {}
    ep_data = []

    if not t2_ep:
        return dm1mo_e, dm1mo_p, lvo_e, lvo_p, ep_data

    for t, comp_p in mf.components.items():
        if not t.startswith('n') or t not in t2_ep:
            continue
        cp = comp_p.mo_coeff
        occ_p = comp_p.mo_occ > 0
        vir_p = comp_p.mo_occ == 0
        nocc_p = numpy.count_nonzero(occ_p)
        nvir_p = numpy.count_nonzero(vir_p)
        if nocc_p == 0 or nvir_p == 0:
            continue

        t2 = t2_ep[t]

        # Eqs. (164)-(167).
        dm1mo_e[numpy.ix_(occ_e, occ_e)] += \
            -2.0 * numpy.einsum('iaIA,jaIA->ij', t2, t2)
        dm1mo_e[numpy.ix_(vir_e, vir_e)] += \
             2.0 * numpy.einsum('iaIA,ibIA->ab', t2, t2)

        dm1mo_t = numpy.zeros((cp.shape[1], cp.shape[1]))
        dm1mo_t[numpy.ix_(occ_p, occ_p)] = \
            -2.0 * numpy.einsum('iaIA,iaJA->IJ', t2, t2)
        dm1mo_t[numpy.ix_(vir_p, vir_p)] = \
             2.0 * numpy.einsum('iaIA,iaIB->AB', t2, t2)
        dm1mo_p[t] = dm1mo_t

        charge_product = comp_e.charge * comp_p.charge
        eri = _EPMOBlocks(comp_e.mol, comp_p.mol, ce, cp,
                          occ_e, vir_e, occ_p, vir_p, charge_product)

        eri_vvov = eri.paIA[nocc_e:]
        eri_ooov = eri.ipIA[:, :nocc_e]
        lvo_e += 4.0 * numpy.einsum('iaIA,caIA->ci', t2, eri_vvov)
        lvo_e -= 4.0 * numpy.einsum('jcIA,jiIA->ci', t2, eri_ooov)

        eri_ovvv = eri.iaPA[:, :, nocc_p:]
        eri_ovII = numpy.diagonal(eri.iaIP[:, :, :, :nocc_p],
                                  axis1=2, axis2=3)
        lvo_t = 4.0 * numpy.einsum('iaIB,iaAB->AI', t2, eri_ovvv)
        lvo_t -= 4.0 * numpy.einsum('iaIA,iaI->AI', t2, eri_ovII)
        lvo_p[t] = lvo_t

        eri.drop_ao_cache()
        ep_data.append((t, t2, eri, charge_product))

    return dm1mo_e, dm1mo_p, lvo_e, lvo_p, ep_data


def _ee_intermediates(mp_e, t2):
    '''Eq. (162), (163), and (169) for the ee part.'''
    mo_coeff = mp_e.mo_coeff
    mo_occ = mp_e.mo_occ
    nmo = mo_coeff.shape[1]
    nocc = numpy.count_nonzero(mo_occ > 0)
    nvir = nmo - nocc
    occ = slice(None, nocc)
    vir = slice(nocc, None)

    t2_bar = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    dm1mo = numpy.zeros((nmo, nmo), dtype=numpy.result_type(t2))
    dm1mo[occ, occ] = -2.0 * numpy.einsum('kiab,kjab->ij', t2, t2_bar)
    dm1mo[vir, vir] = 2.0 * numpy.einsum('ijca,ijcb->ab', t2, t2_bar)

    eri = _EEMOBlocks(mp_e.mol, mo_coeff, nocc)
    lvo = numpy.zeros((nvir, nocc), dtype=numpy.result_type(t2, mo_coeff))
    for c0 in range(0, nvir, eri.vvov_chunk_size()):
        c1 = min(c0 + eri.vvov_chunk_size(), nvir)
        lvo[c0:c1] += 4.0 * numpy.einsum(
            'ijab,cajb->ci', t2_bar, eri.vvov_chunk(c0, c1))
    lvo -= 4.0 * numpy.einsum('kjcb,kmjb->cm', t2_bar, eri.ooov)
    return dm1mo, lvo, t2_bar, eri


def _density_response_rhs(mf, dm1mo):
    dm_ao = {}
    for t, comp in mf.components.items():
        dm_t = dm1mo.get(t)
        if dm_t is None:
            dm_ao[t] = numpy.zeros((comp.mol.nao_nr(), comp.mol.nao_nr()))
        else:
            dm_ao[t] = _mo_density(comp.mo_coeff, dm_t)
    v_ao = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)(dm_ao)

    rhs = {}
    for t, comp in mf.components.items():
        if t not in v_ao:
            continue
        nocc = numpy.count_nonzero(comp.mo_occ > 0)
        rhs[t] = reduce(numpy.dot, (comp.mo_coeff[:, nocc:].T, v_ao[t],
                                    comp.mo_coeff[:, :nocc]))
    return rhs


def _constraint_rhs(mf, dm1mo_p):
    l_f = {}
    for t, dm_t in dm1mo_p.items():
        comp = mf.components[t]
        r_mo = numpy.asarray([reduce(numpy.dot, (comp.mo_coeff.T, r,
                                                 comp.mo_coeff))
                              for r in comp.int1e_r])
        l_f[t] = numpy.einsum('pq,xqp->x', dm_t, r_mo)
    return l_f


def _add_density_response_rhs(lvo_e, lvo_p, density_rhs):
    '''Add the Eq. (178) density-response RHS in CNEO-CPHF units.'''
    lvo_e += 4.0 * density_rhs.get('e', 0)
    for t, rhs_t in density_rhs.items():
        if t.startswith('n'):
            lvo_p[t] = lvo_p.get(t, 0) + 2.0 * rhs_t


def _fixed_occ_response(mf, h1ao, ia, x):
    '''Effective first-order Fock matrix for the fixed occupied-occupied block.

    In field-dependent AO bases, the CPHF variables only contain VO rotations.
    The OO block is fixed by orthonormality, U_ij = -S_ij / 2.  Its induced
    potential belongs to the inhomogeneous RHS of the Z-vector contraction.
    '''
    dm_fixed = {}
    s1_mo = {}
    u_fixed = {}

    for t, comp in mf.components.items():
        nocc = numpy.count_nonzero(comp.mo_occ > 0)
        nmo = comp.mo_coeff.shape[1]
        s1 = _s1mo_for_atom(comp, ia)[x]
        u = numpy.zeros((nmo, nocc), dtype=s1.dtype)
        u[:nocc, :] = -0.5 * s1[:nocc, :nocc]

        fac = 1 if t.startswith('n') else 2
        mocc = comp.mo_coeff[:, :nocc]
        dm = reduce(numpy.dot, (comp.mo_coeff, u * fac, mocc.T))
        dm_fixed[t] = dm + dm.T
        s1_mo[t] = s1
        u_fixed[t] = u

    v_fixed = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)(dm_fixed)

    q_mo = {}
    b_vo = {}
    d_constraint = {}
    for t, comp in mf.components.items():
        nocc = numpy.count_nonzero(comp.mo_occ > 0)
        v = v_fixed.get(t, 0)
        if isinstance(v, numpy.ndarray) and v.ndim == 3:
            v = v[0]
        h1mo = reduce(numpy.dot, (comp.mo_coeff.T,
                                  h1ao[t][ia][x] + v,
                                  comp.mo_coeff))
        q = h1mo - s1_mo[t] * comp.mo_energy
        q_mo[t] = q
        b_vo[t] = q[nocc:, :nocc]

        if t.startswith('n'):
            r_mo = numpy.asarray([reduce(numpy.dot, (comp.mo_coeff.T, r,
                                                     comp.mo_coeff))
                                  for r in comp.int1e_r])
            d_constraint[t] = numpy.einsum('pi,rpi->r',
                                           u_fixed[t],
                                           r_mo[:, :, :nocc])

    return q_mo, s1_mo, b_vo, d_constraint


def _fixed_occ_response_all(mf, h1ao, atmlst):
    nset = len(atmlst) * 3
    q_mo = [[{} for x in range(3)] for ia in atmlst]
    s1_mo = [[{} for x in range(3)] for ia in atmlst]
    b_vo = [[{} for x in range(3)] for ia in atmlst]
    d_constraint = [[{} for x in range(3)] for ia in atmlst]
    dm_fixed = {}
    u_fixed = {}

    for t, comp in mf.components.items():
        nocc = numpy.count_nonzero(comp.mo_occ > 0)
        nmo = comp.mo_coeff.shape[1]
        u_all = numpy.zeros((nset, nmo, nocc),
                            dtype=numpy.result_type(comp.mo_coeff))
        for k, ia in enumerate(atmlst):
            s1_atom = _s1mo_for_atom(comp, ia)
            for x in range(3):
                iset = k * 3 + x
                s1 = s1_atom[x]
                s1_mo[k][x][t] = s1
                u_all[iset, :nocc, :] = -0.5 * s1[:nocc, :nocc]

        fac = 1 if t.startswith('n') else 2
        mocc = comp.mo_coeff[:, :nocc]
        dm = numpy.einsum('up,xpi,vi->xuv',
                          comp.mo_coeff, u_all * fac, mocc)
        dm_fixed[t] = dm + dm.transpose(0, 2, 1)
        u_fixed[t] = u_all

    v_fixed = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)(dm_fixed)

    for t, comp in mf.components.items():
        nocc = numpy.count_nonzero(comp.mo_occ > 0)
        v = v_fixed.get(t, 0)
        if isinstance(v, numpy.ndarray) and v.ndim == 2:
            v = v.reshape(1, *v.shape)
        if t.startswith('n'):
            r_mo = numpy.asarray([reduce(numpy.dot, (comp.mo_coeff.T, r,
                                                     comp.mo_coeff))
                                  for r in comp.int1e_r])
        for k, ia in enumerate(atmlst):
            for x in range(3):
                iset = k * 3 + x
                v1 = v[iset] if isinstance(v, numpy.ndarray) else 0
                h1mo = reduce(numpy.dot, (comp.mo_coeff.T,
                                          h1ao[t][ia][x] + v1,
                                          comp.mo_coeff))
                q = h1mo - s1_mo[k][x][t] * comp.mo_energy
                q_mo[k][x][t] = q
                b_vo[k][x][t] = q[nocc:, :nocc]

                if t.startswith('n'):
                    d_constraint[k][x][t] = numpy.einsum(
                        'pi,rpi->r', u_fixed[t][iset], r_mo[:, :, :nocc])

    return q_mo, s1_mo, b_vo, d_constraint


def _ee_nonresponse_grad(mp_e, t2, t2_bar, eri_mo, q_mo, s1_mo,
                         atmlst):
    mol = mp_e.mol
    mo_coeff = mp_e.mo_coeff
    nocc = numpy.count_nonzero(mp_e.mo_occ > 0)
    nmo = mo_coeff.shape[1]
    co = mo_coeff[:, :nocc]
    cv = mo_coeff[:, nocc:]
    denom_weights = _ee_denom_deriv_weights(t2, t2_bar)
    rotation_terms = _ee_rotation_terms(eri_mo, t2_bar)
    de = numpy.zeros((len(atmlst), 3), dtype=numpy.result_type(t2, mo_coeff))
    de += _ee_deriv_contract(mol, co, cv, t2, atmlst,
                             max(200, mp_e.max_memory-lib.current_memory()[0]))

    for k, ia in enumerate(atmlst):
        for x in range(3):
            q = q_mo[k][x]['e']
            s1 = s1_mo[k][x]['e']
            u = _fill_canonical_mo_response(numpy.zeros((nmo, nocc)),
                                            q, s1, mp_e._scf.mo_energy, nocc)
            e_rot = _ee_rotation_terms_contract(rotation_terms, u)

            eps1 = numpy.diag(q)
            de[k, x] += (e_rot -
                         _ee_denom_deriv_contract(eps1, nocc,
                                                  denom_weights))
    return de.real


def _ep_nonresponse_grad(mf, ep_data, q_mo, s1_mo, atmlst):
    if not ep_data:
        return numpy.zeros((len(atmlst), 3))

    comp_e = mf.components['e']
    mol_e = comp_e.mol
    ce = comp_e.mo_coeff
    occ_e = comp_e.mo_occ > 0
    vir_e = comp_e.mo_occ == 0
    nocc_e = numpy.count_nonzero(occ_e)
    de = numpy.zeros((len(atmlst), 3), dtype=numpy.result_type(ce))

    for t, t2, eri, charge_product in ep_data:
        comp_p = mf.components[t]
        mol_p = comp_p.mol
        cp = comp_p.mo_coeff
        occ_p = comp_p.mo_occ > 0
        vir_p = comp_p.mo_occ == 0
        nocc_p = numpy.count_nonzero(occ_p)
        denom_weights = _ep_denom_deriv_weights(t2)
        rotation_terms = _ep_rotation_terms(eri, t2)

        coe = ce[:, occ_e]
        cve = ce[:, vir_e]
        cop = cp[:, occ_p]
        cvp = cp[:, vir_p]

        for k, ia in enumerate(atmlst):
            dg_bare = _ao_eri_deriv_ep_ovov_cross(
                mol_e, mol_p, coe, cve, cop, cvp, ia, charge_product)
            for x in range(3):
                qe = q_mo[k][x]['e']
                qp = q_mo[k][x][t]
                ue = _fill_canonical_mo_response(
                    numpy.zeros((ce.shape[1], nocc_e)),
                    qe, s1_mo[k][x]['e'], comp_e.mo_energy, nocc_e)
                up = _fill_canonical_mo_response(
                    numpy.zeros((cp.shape[1], nocc_p)),
                    qp, s1_mo[k][x][t], comp_p.mo_energy, nocc_p)

                eps1_e = numpy.diag(qe)
                eps1_p = numpy.diag(qp)
                de[k, x] += (
                    4.0 * numpy.einsum('iaIA,iaIA->', t2, dg_bare[x])
                    + _ep_rotation_terms_contract(rotation_terms, ue, up)
                    - 2.0 * _ep_denom_deriv_contract(
                        eps1_e, eps1_p, nocc_e, nocc_p, denom_weights))

    return de.real


def _solve_z_cneo_adjoint(mf, lvo_e, lvo_p, l_f=None):
    '''Solve Eq. (182), A.T Z = L, in the explicit VO/f space.'''
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    keys = sorted(mo_coeff.keys())
    info = {}
    total = 0
    for t in keys:
        if mo_coeff[t].ndim > 2:
            raise NotImplementedError('CNEO-MP2 gradients for UHF components are not supported')
        nocc = numpy.count_nonzero(mo_occ[t] > 0)
        nmo = mo_coeff[t].shape[1]
        nvir = nmo - nocc
        size = nvir * nocc
        info[t] = {
            'offset': total,
            'nocc': nocc,
            'nmo': nmo,
            'nvir': nvir,
            'size': size,
            'denom': lib.direct_sum('a-i->ai',
                                    mo_energy[t][nocc:], mo_energy[t][:nocc]),
        }
        total += size
    f_offset = {}
    for t in keys:
        if t.startswith('n'):
            f_offset[t] = total
            total += 3

    rhs = numpy.zeros(total)
    rhs[info['e']['offset']:info['e']['offset']+info['e']['size']] = lvo_e.ravel()
    for t, lvo in lvo_p.items():
        if t in info:
            p0 = info[t]['offset']
            rhs[p0:p0+info[t]['size']] = lvo.ravel()
    if l_f:
        for t, lf in l_f.items():
            if t in f_offset:
                rhs[f_offset[t]:f_offset[t]+3] = lf

    fx_full = hessian.gen_vind(mf, mo_coeff, mo_occ)

    def unpack(vec):
        vec = numpy.asarray(vec)
        is_vector = vec.ndim == 1
        if is_vector:
            vec = vec.reshape(1, total)
        nset = vec.shape[0]
        mo1_full = {}
        for t in keys:
            data = info[t]
            block = vec[:, data['offset']:data['offset']+data['size']]
            block = block.reshape(nset, data['nvir'], data['nocc'])
            full = numpy.zeros((nset, data['nmo'], data['nocc']))
            full[:, data['nocc']:, :] = block
            mo1_full[t] = full
        f1 = {}
        for t, p0 in f_offset.items():
            f1[t] = vec[:, p0:p0+3].reshape(nset, 3)
        return mo1_full, f1, is_vector

    def apply_a(vec):
        mo1_full, f1, is_vector = unpack(vec)
        nset = next(iter(mo1_full.values())).shape[0]
        v_full, r = fx_full(mo1_full, f1=f1)
        out = numpy.zeros((nset, total))
        for t in keys:
            data = info[t]
            v = v_full[t].reshape(nset, data['nmo'], data['nocc'])
            vvo = v[:, data['nocc']:, :]
            uvo = mo1_full[t][:, data['nocc']:, :]
            avo = data['denom'] * uvo + vvo
            p0 = data['offset']
            out[:, p0:p0+data['size']] = avo.reshape(nset, data['size'])
        if r is not None:
            for t, rt in r.items():
                if t in f_offset:
                    out[:, f_offset[t]:f_offset[t]+3] = rt.reshape(nset, 3)
        return out.reshape(total) if is_vector else out

    max_memory = max(200, mf.max_memory-lib.current_memory()[0])
    blksize = min(total, 64, _block_size(max_memory * .25, total))
    amat = numpy.empty((total, total))
    for p0 in range(0, total, blksize):
        p1 = min(p0 + blksize, total)
        trial = numpy.zeros((p1-p0, total))
        trial[:, p0:p1] = numpy.eye(p1-p0)
        amat[:, p0:p1] = apply_a(trial).T
    zvec = numpy.linalg.solve(amat.T, rhs)

    z_mo = {}
    for t in keys:
        data = info[t]
        block = zvec[data['offset']:data['offset']+data['size']]
        full = numpy.zeros((data['nmo'], data['nocc']))
        full[data['nocc']:, :] = block.reshape(data['nvir'], data['nocc'])
        z_mo[t] = full
    z_f = {}
    for t, p0 in f_offset.items():
        z_f[t] = zvec[p0:p0+3]
    return z_mo, z_f


def _zvector_mo_gradient(mp_grad, mf, mp_e, t2, atmlst, log, cput0):
    if has_frozen_orbitals(mp_e):
        raise NotImplementedError('CNEO-MP2 gradients do not support '
                                  'frozen electron orbitals')
    if mf.mo_coeff['e'].ndim > 2:
        raise NotImplementedError('CNEO-MP2 gradients for UHF components are not supported')

    atmlst = list(atmlst)

    dm1mo_e, lvo_e, t2_bar, eri_ee_mo = _ee_intermediates(mp_e, t2)
    dm1mo_p = {}
    lvo_p = {}
    ep_data = []
    if mp_grad.base.with_ep:
        ep_dm1mo_e, dm1mo_p, ep_lvo_e, lvo_p, ep_data = \
            _ep_intermediates(mf, mp_grad.base.t2_ep)
        dm1mo_e += ep_dm1mo_e
        lvo_e += ep_lvo_e

    dm_same = {'e': dm1mo_e}
    dm_same.update(dm1mo_p)
    density_rhs = _density_response_rhs(mf, dm_same)
    _add_density_response_rhs(lvo_e, lvo_p, density_rhs)

    l_f = _constraint_rhs(mf, dm1mo_p) if mp_grad.base.with_ep else None
    mo1, z_f = _solve_z_cneo_adjoint(mf, lvo_e, lvo_p, l_f=l_f)

    hessobj = hessian.Hessian(mf)
    hessobj.verbose = mp_grad.verbose
    h1ao = hessobj.make_h1(mf.mo_coeff, mf.mo_occ, None, atmlst, log)

    q_mo, s1_mo, b_vo, d_constraint = \
        _fixed_occ_response_all(mf, h1ao, atmlst)

    de_corr = _ee_nonresponse_grad(mp_e, t2, t2_bar, eri_ee_mo,
                                   q_mo, s1_mo, atmlst)
    if mp_grad.base.with_ep:
        de_corr += _ep_nonresponse_grad(mf, ep_data, q_mo, s1_mo, atmlst)

    for k, ia in enumerate(atmlst):
        for x in range(3):
            for t, comp in mf.components.items():
                nocc = numpy.count_nonzero(comp.mo_occ > 0)
                de_corr[k, x] -= numpy.einsum('ai,ai->',
                                              mo1[t][nocc:, :],
                                              b_vo[k][x][t])
                if t.startswith('n') and t in z_f:
                    de_corr[k, x] -= numpy.einsum('r,r->',
                                                  z_f[t],
                                                  d_constraint[k][x][t])

    mf_grad = mf.nuc_grad_method()
    mf_grad.verbose = mp_grad.verbose
    de = mf_grad.kernel(atmlst=atmlst) + de_corr
    if mp_grad.mol.symmetry:
        de = mp_grad.symmetrize(de, atmlst)

    mp_grad._de_corr = de_corr
    mp_grad.de = de
    mp_grad._finalize()
    log.timer('%s gradients' % mp_grad.base.__class__.__name__, *cput0)
    return mp_grad.de


class Gradients(neo_grad.Gradients):
    '''CNEO-MP2 and CNEO-MP2(ee) gradients'''

    def __init__(self, mp):
        mf = getattr(mp, '_scf', None)
        if mf is None:
            mf = getattr(mp, 'base', None)
        if mf is None:
            raise AttributeError('MP2 object lacks _scf or base attribute required for gradients')
        super().__init__(mf)
        self.base = mp
        self.mp2_grad_slow = getattr(mp, 'mp2_grad_slow', False)
        self._keys = self._keys.union(['mp2_grad_slow'])

    def kernel(self, t2=None, atmlst=None, verbose=None):
        log = logger.new_logger(self, verbose)
        mf = getattr(self.base, '_scf', None)
        if mf is None:
            mf = self.base.base

        cput0 = (logger.process_clock(), logger.perf_counter())

        if t2 is None:
            if getattr(self.base, 't2', None) is None:
                self.base.kernel()
            t2 = self.base.t2
        if self.base.with_ep and getattr(self.base, 't2_ep', None) is None:
            self.base.kernel()
            t2 = self.base.t2

        if atmlst is None:
            atmlst = self.atmlst
            if atmlst is None:
                atmlst = range(self.mol.natm)
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        mp_e = getattr(self.base, 'mp_e', None)
        if mp_e is None:
            mp_e = self.base.mp

        if self.mp2_grad_slow:
            mf_grad = mf.nuc_grad_method()
            mf_grad.verbose = self.verbose
            de_hf = mf_grad.kernel(atmlst=atmlst)
            de_corr_ee = ee_corr_grad(self, mf, mp_e, t2, atmlst, verbose=log)
            de_corr = de_corr_ee
            if self.base.with_ep:
                de_corr_ep = ep_corr_grad(self, mf, mp_e, self.base.t2_ep,
                                          atmlst, verbose=log)
                de_corr = de_corr + de_corr_ep
                self._de_corr_ep = de_corr_ep
            de = de_hf + de_corr
            if self.mol.symmetry:
                de = self.symmetrize(de, atmlst)
            self._de_hf = de_hf
            self._de_corr_ee = de_corr_ee
            self._de_corr = de_corr
            self.de = de
            self._finalize()
            log.timer('%s gradients' % self.base.__class__.__name__, *cput0)
            return self.de

        return _zvector_mo_gradient(self, mf, mp_e, t2, atmlst, log, cput0)
