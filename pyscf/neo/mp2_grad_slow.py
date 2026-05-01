#!/usr/bin/env python
# Slow direct CNEO-MP2(ee) gradient intermediates.

import numpy
from functools import reduce
from pyscf import ao2mo, lib
from pyscf.grad.mp2 import has_frozen_orbitals
from pyscf.lib import logger
from . import hessian


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
