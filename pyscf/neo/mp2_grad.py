#!/usr/bin/env python
# Analytic gradients for CNEO-MP2 and CNEO-MP2(ee)

import numpy
from functools import reduce
from pyscf import lib, ao2mo
from . import hessian, grad as neo_grad
from pyscf.lib import logger
from pyscf.grad.mp2 import has_frozen_orbitals
from .mp2_grad_slow import (ee_corr_grad, ep_corr_grad, _ao_eri_deriv_ovov,
                            _eri_ovov_rotation_deriv, _ep_eri_mo,
                            _ao_eri_deriv_ep_ovov, _ep_ovov_rotation_deriv,
                            _embed_coeff, _fill_canonical_mo_response)


def _mo_density(mo_coeff, dm1mo):
    return reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))


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
        eri_mo = _ep_eri_mo(comp_e.mol, comp_p.mol, ce, cp) * charge_product

        eri_vvov = eri_mo[numpy.ix_(vir_e, vir_e, occ_p, vir_p)]
        eri_ooov = eri_mo[numpy.ix_(occ_e, occ_e, occ_p, vir_p)]
        lvo_e += 4.0 * numpy.einsum('iaIA,caIA->ci', t2, eri_vvov)
        lvo_e -= 4.0 * numpy.einsum('jcIA,jiIA->ci', t2, eri_ooov)

        eri_ovvv = eri_mo[numpy.ix_(occ_e, vir_e, vir_p, vir_p)]
        eri_ovoo = eri_mo[numpy.ix_(occ_e, vir_e, occ_p, occ_p)]
        eri_ovII = numpy.diagonal(eri_ovoo, axis1=2, axis2=3)
        lvo_t = 4.0 * numpy.einsum('iaIB,iaAB->AI', t2, eri_ovvv)
        lvo_t -= 4.0 * numpy.einsum('iaIA,iaI->AI', t2, eri_ovII)
        lvo_p[t] = lvo_t

        ep_data.append((t, t2, eri_mo, charge_product))

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

    eri = mp_e.mol.intor('int2e', aosym='s4')
    eri_mo = ao2mo.incore.full(eri, mo_coeff, compact=False).reshape(
        nmo, nmo, nmo, nmo)
    eri_vvov = eri_mo[vir, vir, occ, vir]
    eri_ooov = eri_mo[occ, occ, occ, vir]
    lvo = 4.0 * numpy.einsum('ijab,cajb->ci', t2_bar, eri_vvov)
    lvo -= 4.0 * numpy.einsum('kjcb,kmjb->cm', t2_bar, eri_ooov)
    return dm1mo, lvo, t2_bar, eri_mo


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


def _ee_nonresponse_grad(mp_e, t2, t2_bar, eri_mo, q_mo, s1_mo,
                         atmlst):
    mol = mp_e.mol
    mo_coeff = mp_e.mo_coeff
    nocc = numpy.count_nonzero(mp_e.mo_occ > 0)
    nmo = mo_coeff.shape[1]
    co = mo_coeff[:, :nocc]
    cv = mo_coeff[:, nocc:]
    eri1 = mol.intor('int2e_ip1', comp=3, aosym='s1')
    eri1 = eri1.reshape(3, mol.nao_nr(), mol.nao_nr(),
                        mol.nao_nr(), mol.nao_nr())
    offset = mol.offset_nr_by_atom()
    t2_weight = t2 * t2_bar
    de = numpy.zeros((len(atmlst), 3), dtype=numpy.result_type(t2, mo_coeff))

    for k, ia in enumerate(atmlst):
        p0, p1 = offset[ia][2:]
        for x in range(3):
            q = q_mo[k][x]['e']
            s1 = s1_mo[k][x]['e']
            u = _fill_canonical_mo_response(numpy.zeros((nmo, nocc)),
                                            q, s1, mp_e._scf.mo_energy, nocc)
            dg = _ao_eri_deriv_ovov(eri1[x], co, cv, p0, p1)
            dg += _eri_ovov_rotation_deriv(eri_mo, u, nocc)

            eps1 = numpy.diag(q)
            ddenom = lib.direct_sum('i+j-a-b->ijab',
                                    eps1[:nocc], eps1[:nocc],
                                    eps1[nocc:], eps1[nocc:])
            de[k, x] = (2.0 * numpy.einsum('ijab,ijab->', t2_bar, dg)
                        - numpy.einsum('ijab,ijab->', t2_weight, ddenom))
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
    nao_e = mol_e.nao_nr()
    de = numpy.zeros((len(atmlst), 3), dtype=numpy.result_type(ce))

    for t, t2, eri_mo, charge_product in ep_data:
        comp_p = mf.components[t]
        mol_p = comp_p.mol
        cp = comp_p.mo_coeff
        occ_p = comp_p.mo_occ > 0
        vir_p = comp_p.mo_occ == 0
        nocc_p = numpy.count_nonzero(occ_p)

        mol_tot = mol_e + mol_p
        nao_tot = mol_tot.nao_nr()
        coe = _embed_coeff(ce[:, occ_e], nao_tot, 0)
        cve = _embed_coeff(ce[:, vir_e], nao_tot, 0)
        cop = _embed_coeff(cp[:, occ_p], nao_tot, nao_e)
        cvp = _embed_coeff(cp[:, vir_p], nao_tot, nao_e)
        eri1 = mol_tot.intor('int2e_ip1', comp=3, aosym='s1')
        eri1 = eri1.reshape(3, nao_tot, nao_tot, nao_tot, nao_tot)
        offset_e = mol_e.offset_nr_by_atom()
        offset_p = mol_p.offset_nr_by_atom()

        for k, ia in enumerate(atmlst):
            p0e, p1e = offset_e[ia][2:]
            p0p, p1p = offset_p[ia][2:]
            p0p += nao_e
            p1p += nao_e
            for x in range(3):
                qe = q_mo[k][x]['e']
                qp = q_mo[k][x][t]
                ue = _fill_canonical_mo_response(
                    numpy.zeros((ce.shape[1], nocc_e)),
                    qe, s1_mo[k][x]['e'], comp_e.mo_energy, nocc_e)
                up = _fill_canonical_mo_response(
                    numpy.zeros((cp.shape[1], nocc_p)),
                    qp, s1_mo[k][x][t], comp_p.mo_energy, nocc_p)

                dg = _ao_eri_deriv_ep_ovov(eri1[x], coe, cve, cop, cvp,
                                           p0e, p1e, p0p, p1p,
                                           charge_product)
                dg += _ep_ovov_rotation_deriv(eri_mo, ue, up,
                                              nocc_e, nocc_p)

                eps1_e = numpy.diag(qe)
                eps1_p = numpy.diag(qp)
                ddenom = (eps1_e[:nocc_e, None, None, None]
                          + eps1_p[None, None, :nocc_p, None]
                          - eps1_e[None, nocc_e:, None, None]
                          - eps1_p[None, None, None, nocc_p:])
                de[k, x] += (
                    4.0 * numpy.einsum('iaIA,iaIA->', t2, dg)
                    - 2.0 * numpy.einsum('iaIA,iaIA->', t2 * t2, ddenom))

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
        mo1_full = {}
        for t in keys:
            data = info[t]
            block = vec[data['offset']:data['offset']+data['size']]
            block = block.reshape(data['nvir'], data['nocc'])
            full = numpy.zeros((1, data['nmo'], data['nocc']))
            full[0, data['nocc']:, :] = block
            mo1_full[t] = full
        f1 = {}
        for t, p0 in f_offset.items():
            f1[t] = vec[p0:p0+3].reshape(1, 3)
        return mo1_full, f1

    def apply_a(vec):
        mo1_full, f1 = unpack(vec)
        v_full, r = fx_full(mo1_full, f1=f1)
        out = numpy.zeros(total)
        for t in keys:
            data = info[t]
            v = v_full[t].reshape(1, data['nmo'], data['nocc'])
            vvo = v[0, data['nocc']:, :]
            uvo = mo1_full[t][0, data['nocc']:, :]
            avo = data['denom'] * uvo + vvo
            p0 = data['offset']
            out[p0:p0+data['size']] = avo.ravel()
        if r is not None:
            for t, rt in r.items():
                if t in f_offset:
                    out[f_offset[t]:f_offset[t]+3] = rt.reshape(3)
        return out

    eye = numpy.eye(total)
    amat = numpy.asarray([apply_a(eye[:, i]) for i in range(total)]).T
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

    q_mo = []
    s1_mo = []
    b_vo = []
    d_constraint = []
    for ia in atmlst:
        q_atom = []
        s_atom = []
        b_atom = []
        d_atom = []
        for x in range(3):
            qx, sx, bx, dx = _fixed_occ_response(mf, h1ao, ia, x)
            q_atom.append(qx)
            s_atom.append(sx)
            b_atom.append(bx)
            d_atom.append(dx)
        q_mo.append(q_atom)
        s1_mo.append(s_atom)
        b_vo.append(b_atom)
        d_constraint.append(d_atom)

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
