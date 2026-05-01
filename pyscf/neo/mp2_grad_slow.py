#!/usr/bin/env python
# Slow direct CNEO-MP2 gradients

import numpy
from functools import reduce
from pyscf import ao2mo, lib
from pyscf.grad.mp2 import has_frozen_orbitals
from pyscf.lib import logger
from . import cphf, hessian


def _get_hessian_object(mf):
    if hasattr(mf, 'Hessian'):
        return mf.Hessian()
    return hessian.Hessian(mf)


def _s1ao_for_atom(mol, s1a, ia):
    nao = mol.nao_nr()
    p0, p1 = mol.aoslice_by_atom()[ia][2:]
    s1ao = numpy.zeros((3, nao, nao), dtype=s1a.dtype)
    s1ao[:, p0:p1] += s1a[:, p0:p1]
    s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
    return s1ao


def _component_mo1_from_ao(comp, mo1_ao):
    s0 = comp.get_ovlp()
    return numpy.einsum('up,uv,xvi->xpi', comp.mo_coeff, s0, mo1_ao)


def _make_cneo_response_dm1(mf, mo1_ao, atmlst):
    dm1 = {}
    for t, comp in mf.components.items():
        if comp.mo_coeff.ndim > 2:
            raise NotImplementedError('CNEO-MP2 gradients for UHF components are not supported')
        occidx = comp.mo_occ > 0
        mocc = comp.mo_coeff[:, occidx]
        c1 = numpy.asarray([mo1_ao[t][ia][x] for ia in atmlst for x in range(3)])
        fac = 1 if t.startswith('n') else 2
        dm = numpy.einsum('xui,vi->xuv', c1 * fac, mocc)
        dm1[t] = dm + dm.transpose(0, 2, 1)
    return dm1


def _make_cneo_response_dm1_mo(mf, mo1_mo):
    dm1 = {}
    for t, comp in mf.components.items():
        if comp.mo_coeff.ndim > 2:
            raise NotImplementedError('CNEO-MP2 gradients for UHF components are not supported')
        occidx = comp.mo_occ > 0
        mocc = comp.mo_coeff[:, occidx]
        fac = 1 if t.startswith('n') else 2
        dm = numpy.einsum('up,xpi,vi->xuv', comp.mo_coeff, mo1_mo[t] * fac, mocc)
        dm1[t] = dm + dm.transpose(0, 2, 1)
    return dm1


def _s1mo_for_atom(comp, ia):
    s1a = -comp.mol.intor('int1e_ipovlp', comp=3)
    s1ao = _s1ao_for_atom(comp.mol, s1a, ia)
    return numpy.asarray([reduce(numpy.dot, (comp.mo_coeff.T, s1ao[x],
                                             comp.mo_coeff))
                          for x in range(3)])


def _solve_cneo_cphf_context(mp_grad, mf, atmlst, verbose=None):
    log = logger.new_logger(mp_grad, verbose)
    atmlst = list(atmlst)
    hessobj = _get_hessian_object(mf)
    hessobj.verbose = mp_grad.verbose
    h1ao = hessobj.make_h1(mf.mo_coeff, mf.mo_occ, None, atmlst, log)

    h1vo = {}
    s1vo = {}
    for t, comp in mf.components.items():
        if comp.mo_coeff.ndim > 2:
            raise NotImplementedError('CNEO-MP2 gradients for UHF components are not supported')
        occidx = comp.mo_occ > 0
        mocc = comp.mo_coeff[:, occidx]
        h1_blocks = []
        s1_blocks = []
        for ia in atmlst:
            s1mo = _s1mo_for_atom(comp, ia)
            h1mo = numpy.asarray([reduce(numpy.dot, (comp.mo_coeff.T,
                                                     h1ao[t][ia][x], mocc))
                                  for x in range(3)])
            h1_blocks.append(h1mo)
            s1_blocks.append(s1mo[:, :, occidx])
        h1vo[t] = numpy.vstack(h1_blocks)
        s1vo[t] = numpy.vstack(s1_blocks)

    fx = hessian.gen_vind(mf, mf.mo_coeff, mf.mo_occ)
    tol = getattr(mf, 'conv_tol_cpscf', 1e-9) * max(1, len(atmlst))
    mo1_mo, _, f1 = cphf.solve(
        fx, mf.mo_energy, mf.mo_occ, h1vo, s1vo,
        with_f1=True, verbose=log, max_cycle=hessobj.max_cycle,
        level_shift=hessobj.level_shift, tol=tol)

    dm1 = _make_cneo_response_dm1_mo(mf, mo1_mo)
    v1ao = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)(dm1)

    q_mo = {}
    s1_mo = {}
    for t, comp in mf.components.items():
        q_mo[t] = {}
        s1_mo[t] = {}
        for k, ia in enumerate(atmlst):
            q_blk = []
            s1_blk = _s1mo_for_atom(comp, ia)
            for x in range(3):
                iset = k * 3 + x
                f1ao = h1ao[t][ia][x] + v1ao[t][iset]
                if t.startswith('n') and f1 is not None and t in f1:
                    f1ao = f1ao + numpy.einsum('r,ruv->uv',
                                               f1[t][iset], comp.int1e_r)
                f1mo = reduce(numpy.dot, (comp.mo_coeff.T, f1ao, comp.mo_coeff))
                q_blk.append(f1mo - s1_blk[x] * comp.mo_energy)
            q_mo[t][ia] = numpy.asarray(q_blk)
            s1_mo[t][ia] = s1_blk

    return {
        'atmlst': atmlst,
        'h1ao': h1ao,
        'mo1_mo': mo1_mo,
        'q_mo': q_mo,
        's1_mo': s1_mo,
        'f1': f1,
    }


def _ao_eri_deriv_ovov(eri1, co, cv, p0, p1):
    co_atm = numpy.zeros_like(co)
    cv_atm = numpy.zeros_like(cv)
    co_atm[p0:p1] = co[p0:p1]
    cv_atm[p0:p1] = cv[p0:p1]
    nocc = co.shape[1]
    nvir = cv.shape[1]

    def transform(coeffs, shape):
        return ao2mo.incore.general(eri1, coeffs, compact=False).reshape(shape)

    d_i = transform((co_atm, cv, co, cv), (nocc, nvir, nocc, nvir))
    d_a = transform((cv_atm, co, co, cv), (nvir, nocc, nocc, nvir))
    dovov = d_i.copy()
    dovov += d_a.transpose(1, 0, 2, 3)
    dovov += d_i.transpose(2, 3, 0, 1)
    dovov += d_a.transpose(2, 3, 1, 0)
    return -dovov.transpose(0, 2, 1, 3)


def _fill_canonical_mo_response(u_occ, q, s1mo, mo_energy, nocc):
    nmo = mo_energy.size
    nvir = nmo - nocc
    u = numpy.zeros((nmo, nmo), dtype=numpy.result_type(u_occ, q, s1mo))
    e_occ = mo_energy[:nocc]
    e_vir = mo_energy[nocc:]

    u[nocc:, :nocc] = u_occ[nocc:, :nocc]
    u[:nocc, nocc:] = -u[nocc:, :nocc].T - s1mo[:nocc, nocc:]

    e_ij = lib.direct_sum('j-i->ij', e_occ, e_occ)
    mask = abs(e_ij) > 1e-12
    uoo = u[:nocc, :nocc]
    uoo[mask] = q[:nocc, :nocc][mask] / e_ij[mask]
    uoo[~mask] = -0.5 * s1mo[:nocc, :nocc][~mask]

    e_ab = lib.direct_sum('b-a->ab', e_vir, e_vir)
    mask = abs(e_ab) > 1e-12
    uvv = u[nocc:, nocc:]
    uvv[mask] = q[nocc:, nocc:][mask] / e_ab[mask]
    uvv[~mask] = -0.5 * s1mo[nocc:, nocc:][~mask]
    if nvir:
        diag = numpy.diag_indices(nvir)
        uvv[diag] = -0.5 * s1mo[nocc:, nocc:][diag]
    if nocc:
        diag = numpy.diag_indices(nocc)
        uoo[diag] = -0.5 * s1mo[:nocc, :nocc][diag]
    return u


def _eri_ovov_rotation_deriv(eri_mo, u, nocc):
    uo = u[:, :nocc]
    uv = u[:, nocc:]
    o = slice(None, nocc)
    v = slice(nocc, None)
    d = numpy.einsum('pi,pajb->iajb', uo, eri_mo[:, v, o, v])
    d += numpy.einsum('pa,ipjb->iajb', uv, eri_mo[o, :, o, v])
    d += numpy.einsum('pj,iapb->iajb', uo, eri_mo[o, v, :, v])
    d += numpy.einsum('pb,iajp->iajb', uv, eri_mo[o, v, o, :])
    return d.transpose(0, 2, 1, 3)


def _embed_coeff(coeff, nao_tot, p0):
    out = numpy.zeros((nao_tot, coeff.shape[1]), dtype=coeff.dtype)
    out[p0:p0 + coeff.shape[0]] = coeff
    return out


def _ep_eri_mo(mol_e, mol_n, mo_coeff_e, mo_coeff_n):
    nao_e = mol_e.nao_nr()
    nao_n = mol_n.nao_nr()
    nao_tot = nao_e + nao_n
    ce = _embed_coeff(mo_coeff_e, nao_tot, 0)
    cn = _embed_coeff(mo_coeff_n, nao_tot, nao_e)
    eri = (mol_e + mol_n).intor('int2e', aosym='s4')
    return ao2mo.incore.general(eri, (ce, ce, cn, cn),
                                compact=False).reshape(
                                    mo_coeff_e.shape[1], mo_coeff_e.shape[1],
                                    mo_coeff_n.shape[1], mo_coeff_n.shape[1])


def _ao_eri_deriv_ep_ovov(eri1, coe, cve, con, cvn,
                          p0e, p1e, p0n, p1n, charge_product):
    nocc_e = coe.shape[1]
    nvir_e = cve.shape[1]
    nocc_n = con.shape[1]
    nvir_n = cvn.shape[1]
    dovov = numpy.zeros((nocc_e, nvir_e, nocc_n, nvir_n),
                        dtype=numpy.result_type(eri1, coe, con))

    def transform(coeffs, shape):
        return ao2mo.incore.general(eri1, coeffs, compact=False).reshape(shape)

    if p1e > p0e:
        coe_atm = numpy.zeros_like(coe)
        cve_atm = numpy.zeros_like(cve)
        coe_atm[p0e:p1e] = coe[p0e:p1e]
        cve_atm[p0e:p1e] = cve[p0e:p1e]
        d_i = transform((coe_atm, cve, con, cvn),
                        (nocc_e, nvir_e, nocc_n, nvir_n))
        d_a = transform((cve_atm, coe, con, cvn),
                        (nvir_e, nocc_e, nocc_n, nvir_n))
        dovov += d_i
        dovov += d_a.transpose(1, 0, 2, 3)

    if p1n > p0n:
        con_atm = numpy.zeros_like(con)
        cvn_atm = numpy.zeros_like(cvn)
        con_atm[p0n:p1n] = con[p0n:p1n]
        cvn_atm[p0n:p1n] = cvn[p0n:p1n]
        d_I = transform((con_atm, cvn, coe, cve),
                        (nocc_n, nvir_n, nocc_e, nvir_e))
        d_A = transform((cvn_atm, con, coe, cve),
                        (nvir_n, nocc_n, nocc_e, nvir_e))
        dovov += d_I.transpose(2, 3, 0, 1)
        dovov += d_A.transpose(2, 3, 1, 0)

    return -charge_product * dovov


def _ep_ovov_rotation_deriv(eri_mo, ue, up, nocc_e, nocc_n):
    oe = slice(None, nocc_e)
    ve = slice(nocc_e, None)
    on = slice(None, nocc_n)
    vn = slice(nocc_n, None)
    d = numpy.einsum('pi,paIA->iaIA', ue[:, oe], eri_mo[:, ve, on, vn])
    d += numpy.einsum('pa,ipIA->iaIA', ue[:, ve], eri_mo[oe, :, on, vn])
    d += numpy.einsum('PI,iaPA->iaIA', up[:, on], eri_mo[oe, ve, :, vn])
    d += numpy.einsum('PA,iaIP->iaIA', up[:, vn], eri_mo[oe, ve, on, :])
    return d


def ee_corr_grad(mp_grad, mf, mp_e, t2, atmlst, verbose=None):
    '''CNEO-MP2(ee) correlation gradient'''
    log = logger.new_logger(mp_grad, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    if has_frozen_orbitals(mp_e):
        raise NotImplementedError('CNEO-MP2(ee) gradients do not support '
                                  'frozen electron orbitals')
    if mf.mo_coeff['e'].ndim > 2:
        raise NotImplementedError('CNEO-MP2 gradients for UHF components are not supported')

    hessobj = _get_hessian_object(mf)
    hessobj.verbose = mp_grad.verbose
    h1ao = hessobj.make_h1(mf.mo_coeff, mf.mo_occ, None, atmlst, log)
    mo1_ao, _ = hessobj.solve_mo1(mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao,
                                  None, atmlst, mf.max_memory, log)
    dm1 = _make_cneo_response_dm1(mf, mo1_ao, atmlst)
    v1ao = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)(dm1)

    mol = mp_e.mol
    mo_coeff = mp_e.mo_coeff
    mo_energy = mp_e._scf.mo_energy
    nmo = mo_coeff.shape[1]
    nocc = numpy.count_nonzero(mp_e.mo_occ > 0)
    co = mo_coeff[:, :nocc]
    cv = mo_coeff[:, nocc:]

    eri_ao = mol.intor('int2e', aosym='s4')
    eri_mo = ao2mo.incore.full(eri_ao, mo_coeff, compact=False).reshape(nmo, nmo, nmo, nmo)
    eri1 = mol.intor('int2e_ip1', comp=3, aosym='s1')
    eri1 = eri1.reshape(3, mol.nao_nr(), mol.nao_nr(), mol.nao_nr(), mol.nao_nr())

    t2_bar = 2 * t2 - t2.transpose(0, 1, 3, 2)
    t2_weight = t2 * t2_bar
    s1a_e = -mol.intor('int1e_ipovlp', comp=3)
    offsetdic = mol.offset_nr_by_atom()

    mo1_mo_e = {}
    comp_e = mf.components['e']
    for ia in atmlst:
        mo1_mo_e[ia] = _component_mo1_from_ao(comp_e, mo1_ao['e'][ia])

    de = numpy.zeros((len(atmlst), 3), dtype=numpy.result_type(t2, mo_coeff))
    for k, ia in enumerate(atmlst):
        p0, p1 = offsetdic[ia][2:]
        s1ao = _s1ao_for_atom(mol, s1a_e, ia)
        for x in range(3):
            iset = k * 3 + x
            s1mo = reduce(numpy.dot, (mo_coeff.T, s1ao[x], mo_coeff))
            f1mo = reduce(numpy.dot, (mo_coeff.T, h1ao['e'][ia][x] + v1ao['e'][iset], mo_coeff))
            q = f1mo - s1mo * mo_energy
            u = _fill_canonical_mo_response(mo1_mo_e[ia][x], q, s1mo, mo_energy, nocc)

            dg = _ao_eri_deriv_ovov(eri1[x], co, cv, p0, p1)
            dg += _eri_ovov_rotation_deriv(eri_mo, u, nocc)

            eps1 = numpy.diag(q)
            ddenom = lib.direct_sum('i+j-a-b->ijab',
                                    eps1[:nocc], eps1[:nocc],
                                    eps1[nocc:], eps1[nocc:])
            de[k, x] = (2 * numpy.einsum('ijab,ijab->', t2_bar, dg)
                        - numpy.einsum('ijab,ijab->', t2_weight, ddenom))

    log.timer_debug1('CNEO-MP2(ee) correlation gradient', *time0)
    return de.real


def ep_corr_grad(mp_grad, mf, mp_e, t2_ep, atmlst, verbose=None):
    '''
    electron-proton correlation gradient
    '''
    log = logger.new_logger(mp_grad, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    if has_frozen_orbitals(mp_e):
        raise NotImplementedError('CNEO-MP2 gradients do not support '
                                  'frozen electron orbitals')
    if mf.mo_coeff['e'].ndim > 2:
        raise NotImplementedError('CNEO-MP2 gradients for UHF components are not supported')
    if not t2_ep:
        return numpy.zeros((len(atmlst), 3))

    atmlst = list(atmlst)
    ctx = _solve_cneo_cphf_context(mp_grad, mf, atmlst, verbose=log)
    mo1_mo = ctx['mo1_mo']
    q_mo = ctx['q_mo']
    s1_mo = ctx['s1_mo']

    comp_e = mf.components['e']
    mol_e = comp_e.mol
    ce = comp_e.mo_coeff
    eps_e = comp_e.mo_energy
    occidx_e = comp_e.mo_occ > 0
    viridx_e = comp_e.mo_occ == 0
    nocc_e = numpy.count_nonzero(occidx_e)
    if not occidx_e.any() or not viridx_e.any():
        return numpy.zeros((len(atmlst), 3))

    de = numpy.zeros((len(atmlst), 3), dtype=numpy.result_type(ce))
    nao_e = mol_e.nao_nr()

    for t, comp_n in mf.components.items():
        if not t.startswith('n') or t not in t2_ep:
            continue
        if comp_n.mo_coeff.ndim > 2:
            raise NotImplementedError('CNEO-MP2 gradients for UHF components are not supported')

        mo_occ_n = comp_n.mo_occ
        occidx_n = mo_occ_n > 0
        viridx_n = mo_occ_n == 0
        if not occidx_n.any() or not viridx_n.any():
            continue

        cn = comp_n.mo_coeff
        eps_n = comp_n.mo_energy
        nocc_n = numpy.count_nonzero(occidx_n)
        t2 = t2_ep[t]

        charge_product = comp_e.charge * comp_n.charge
        eri_mo = _ep_eri_mo(mol_e, comp_n.mol, ce, cn) * charge_product

        mol_tot = mol_e + comp_n.mol
        nao_tot = mol_tot.nao_nr()
        coe = _embed_coeff(ce[:, occidx_e], nao_tot, 0)
        cve = _embed_coeff(ce[:, viridx_e], nao_tot, 0)
        con = _embed_coeff(cn[:, occidx_n], nao_tot, nao_e)
        cvn = _embed_coeff(cn[:, viridx_n], nao_tot, nao_e)
        eri1 = mol_tot.intor('int2e_ip1', comp=3, aosym='s1')
        eri1 = eri1.reshape(3, nao_tot, nao_tot, nao_tot, nao_tot)

        offset_e = mol_e.offset_nr_by_atom()
        offset_n = comp_n.mol.offset_nr_by_atom()

        for k, ia in enumerate(atmlst):
            p0e, p1e = offset_e[ia][2:]
            p0n, p1n = offset_n[ia][2:]
            p0n += nao_e
            p1n += nao_e
            for x in range(3):
                iset = k * 3 + x
                qe = q_mo['e'][ia][x]
                qn = q_mo[t][ia][x]
                ue = _fill_canonical_mo_response(mo1_mo['e'][iset], qe,
                                                 s1_mo['e'][ia][x],
                                                 eps_e, nocc_e)
                up = _fill_canonical_mo_response(mo1_mo[t][iset], qn,
                                                 s1_mo[t][ia][x],
                                                 eps_n, nocc_n)

                dg = _ao_eri_deriv_ep_ovov(eri1[x], coe, cve, con, cvn,
                                           p0e, p1e, p0n, p1n,
                                           charge_product)
                dg += _ep_ovov_rotation_deriv(eri_mo, ue, up,
                                              nocc_e, nocc_n)

                eps1_e = numpy.diag(qe)
                eps1_n = numpy.diag(qn)
                ddenom = (eps1_e[:nocc_e, None, None, None]
                          + eps1_n[None, None, :nocc_n, None]
                          - eps1_e[None, nocc_e:, None, None]
                          - eps1_n[None, None, None, nocc_n:])
                de[k, x] += (
                    4 * numpy.einsum('iaIA,iaIA->', t2, dg)
                    - 2 * numpy.einsum('iaIA,iaIA->', t2 * t2, ddenom))

    log.timer_debug1('CNEO-MP2(ep) correlation gradient', *time0)
    return de.real
