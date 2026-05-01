#!/usr/bin/env python
# Analytic gradients for CNEO-MP2(ee)

import numpy
from functools import reduce
from pyscf import lib
from pyscf.grad import rhf as rhf_grad
from . import hessian, cphf, grad as neo_grad
from pyscf.mp import mp2
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.grad.mp2 import _index_frozen_active, has_frozen_orbitals, _shell_prange
from .mp2_grad_slow import ee_corr_grad, ep_corr_grad


def _mo1_to_dm1(mo1, mo_occ):
    '''Build response density matrix in MO basis from mo1 (vo block).

    mo1 shape: (nmo, nocc) for restricted components.
    '''
    occidx = mo_occ > 0
    viridx = ~occidx
    nmo = mo_occ.size
    dm1mo = numpy.zeros((nmo, nmo))
    if mo1 is None:
        return dm1mo
    occ = numpy.where(occidx)[0]
    vir = numpy.where(viridx)[0]
    dm1mo[numpy.ix_(vir, occ)] = mo1[vir]
    dm1mo[numpy.ix_(occ, vir)] = mo1[vir].T
    return dm1mo


def _make_zeta(mo_energy, mo_coeff, mo_occ, dm1mo):
    '''Energy-weighted density contribution from dm1mo.'''
    nocc = numpy.count_nonzero(mo_occ > 0)
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    return reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))


def _get_self_veff(comp, mol, dm):
    '''Get self-type veff (exclude cached inter-component Coulomb).'''
    saved_vint = getattr(comp, '_vint', None)
    comp._vint = None
    try:
        veff = comp.get_veff(mol, dm)
    finally:
        comp._vint = saved_vint
    return veff


def _solve_cp_cneo(mf, Xvo, Lp, max_cycle=50, tol=1e-9):
    '''Solve CNEO Z-vector equations'''
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff

    h1 = {}
    for t in mo_coeff.keys():
        if mo_coeff[t].ndim > 2:
            raise NotImplementedError('CNEO-MP2(ee) gradients for UHF components are not supported')
        nocc = numpy.count_nonzero(mo_occ[t] > 0)
        nmo = mo_coeff[t].shape[1]
        nvir = nmo - nocc
        h1_t = numpy.zeros((nvir, nocc))
        if t == 'e':
            h1_t[:, :] = Xvo
        elif t in Lp:
            h1_t[:, :] = Lp[t]
        h1[t] = h1_t

    fx_full = hessian.gen_vind(mf, mo_coeff, mo_occ)

    def fx(mo1_vo, f1=None):
        '''Wrapper to expand VO blocks to full MO and compress response back to VO.'''
        mo1_full = {}
        for t, comp in mo1_vo.items():
            nocc = numpy.count_nonzero(mo_occ[t] > 0)
            nmo = mo_coeff[t].shape[1]
            nvir = nmo - nocc
            comp = comp.reshape(-1, nvir, nocc)
            full = numpy.zeros((comp.shape[0], nmo, nocc))
            full[:, nocc:, :] = comp
            mo1_full[t] = full
        v1_full, r1 = fx_full(mo1_full, f1=f1)
        v1_vo = {}
        for t, comp in v1_full.items():
            nocc = numpy.count_nonzero(mo_occ[t] > 0)
            nmo = mo_coeff[t].shape[1]
            nvir = nmo - nocc
            comp = comp.reshape(-1, nmo, nocc)
            v1_vo[t] = comp[:, nocc:, :].reshape(-1, nvir*nocc)
        return v1_vo, r1

    mo1, _, f1 = cphf.solve_nos1(
        fx, mo_energy, mo_occ, h1,
        with_f1=True, max_cycle=max_cycle, tol=tol
    )
    # Expand VO blocks to full MO x occ for downstream density builders
    for t in mo1:
        nocc = numpy.count_nonzero(mo_occ[t] > 0)
        nmo = mo_coeff[t].shape[1]
        nvir = nmo - nocc
        mo1_full = numpy.zeros((nmo, nocc))
        mo1_full[nocc:, :] = mo1[t].reshape(nvir, nocc)
        mo1[t] = mo1_full
    return mo1, f1


class Gradients(neo_grad.Gradients):
    '''CNEO-MP2(ee) gradients'''

    def __init__(self, mp):
        mf = getattr(mp, '_scf', None)
        if mf is None:
            mf = getattr(mp, 'base', None)
        if mf is None:
            raise AttributeError('MP2 object lacks _scf or base attribute required for gradients')
        super().__init__(mf)
        self.base = mp
        self.mp2_grad_slow = getattr(mp, 'mp2_grad_slow', True)
        self._keys = self._keys.union(['mp2_grad_slow'])

    def kernel(self, t2=None, atmlst=None, verbose=None):
        log = logger.new_logger(self, verbose)
        mf = getattr(self.base, '_scf', None)
        if mf is None:
            mf = self.base.base

        cput0 = (logger.process_clock(), logger.perf_counter())

        if self.base.with_ep and not self.mp2_grad_slow:
            raise NotImplementedError('CNEO-MP2 gradients are not supported yet; '
                                      'only the slow CNEO-CPHF implementation is available')

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
        mol_e = mp_e.mol

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

        # ===== 1) MP2 1- and 2-pdm intermediates for electrons =====
        d1 = mp2._gamma1_intermediates(mp_e, t2)
        doo, dvv = d1

        with_frozen = has_frozen_orbitals(mp_e)
        OA, VA, OF, VF = _index_frozen_active(mp_e.get_frozen_mask(), mp_e.mo_occ)
        orbo = mp_e.mo_coeff[:, OA]
        orbv = mp_e.mo_coeff[:, VA]
        nao, nocc_act = orbo.shape
        nvir_act = orbv.shape[1]

        part_dm2 = _ao2mo.nr_e2(t2.reshape(nocc_act**2, nvir_act**2),
                                numpy.asarray(orbv.T, order='F'), (0, nao, 0, nao),
                                's1', 's1').reshape(nocc_act, nocc_act, nao, nao)
        part_dm2 = (part_dm2.transpose(0, 2, 3, 1) * 4 -
                    part_dm2.transpose(0, 3, 2, 1) * 2)

        hf_dm1_e = mp_e._scf.make_rdm1(mp_e.mo_coeff, mp_e.mo_occ)

        if atmlst is None:
            atmlst = range(mol_e.natm)
        offsetdic = mol_e.offset_nr_by_atom()
        diagidx = numpy.arange(nao)
        diagidx = diagidx*(diagidx+1)//2 + diagidx
        de_e = numpy.zeros((len(atmlst), 3))
        Imat = numpy.zeros((nao, nao))
        fdm2 = lib.H5TmpFile()
        vhf1 = fdm2.create_dataset('vhf1', (len(atmlst), 3, nao, nao), 'f8')

        max_memory = max(0, mp_e.max_memory - lib.current_memory()[0])
        blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))

        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = offsetdic[ia]
            ip1 = p0
            vhf = numpy.zeros((3, nao, nao))
            for b0, b1, nf in _shell_prange(mol_e, shl0, shl1, blksize):
                ip0, ip1 = ip1, ip1 + nf
                dm2buf = lib.einsum('pi,iqrj->pqrj', orbo[ip0:ip1], part_dm2)
                dm2buf += lib.einsum('qi,iprj->pqrj', orbo, part_dm2[:, ip0:ip1])
                dm2buf = lib.einsum('pqrj,sj->pqrs', dm2buf, orbo)
                dm2buf = dm2buf + dm2buf.transpose(0, 1, 3, 2)
                dm2buf = lib.pack_tril(dm2buf.reshape(-1, nao, nao)).reshape(nf, nao, -1)
                dm2buf[:, :, diagidx] *= .5

                shls_slice = (b0, b1, 0, mol_e.nbas, 0, mol_e.nbas, 0, mol_e.nbas)
                eri0 = mol_e.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
                Imat += lib.einsum('ipx,iqx->pq', eri0.reshape(nf, nao, -1), dm2buf)
                eri0 = None

                eri1 = mol_e.intor('int2e_ip1', comp=3, aosym='s2kl',
                                 shls_slice=shls_slice).reshape(3, nf, nao, -1)
                de_e[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2
                dm2buf = None
                for i in range(3):
                    eri1tmp = lib.unpack_tril(eri1[i].reshape(nf*nao, -1))
                    eri1tmp = eri1tmp.reshape(nf, nao, nao, nao)
                    vhf[i] += numpy.einsum('ijkl,ij->kl', eri1tmp, hf_dm1_e[ip0:ip1])
                    vhf[i] -= numpy.einsum('ijkl,il->kj', eri1tmp, hf_dm1_e[ip0:ip1]) * .5
                    vhf[i, ip0:ip1] += numpy.einsum('ijkl,kl->ij', eri1tmp, hf_dm1_e)
                    vhf[i, ip0:ip1] -= numpy.einsum('ijkl,jk->il', eri1tmp, hf_dm1_e) * .5
                eri1 = eri1tmp = None
            vhf1[k] = vhf

        # ===== 2) Build MP2 unrelaxed density (P2,e) =====
        mo_coeff = mp_e.mo_coeff
        mo_energy = mp_e._scf.mo_energy
        nocc = numpy.count_nonzero(mp_e.mo_occ > 0)
        Imat = reduce(numpy.dot, (mo_coeff.T, Imat, mp_e._scf.get_ovlp(), mo_coeff)) * -1

        dm1mo = numpy.zeros((mo_coeff.shape[1], mo_coeff.shape[1]))
        if with_frozen:
            dm1mo[OA[:, None], OA] = doo + doo.T
            dm1mo[VA[:, None], VA] = dvv + dvv.T
        else:
            dm1mo[:nocc, :nocc] = doo + doo.T
            dm1mo[nocc:, nocc:] = dvv + dvv.T
        # Difference-density diagonal P^(2),e_kk = dE(2)/dε_k for drag force
        # Build in full MO basis; frozen orbitals have zero contribution.
        pdiag = numpy.zeros(mo_coeff.shape[1], dtype=doo.dtype)
        frozen_mask = mp_e.get_frozen_mask()
        occ_mask = (mp_e.mo_occ > 0) & frozen_mask
        vir_mask = (mp_e.mo_occ == 0) & frozen_mask
        # doo/dvv are in the active MO ordering (frozen removed).
        # For the proton drag term, these diagonal intermediates already refer
        # to the spatial-orbital energy response and should not be doubled again.
        pdiag[occ_mask] = numpy.diag(doo)
        pdiag[vir_mask] = numpy.diag(dvv)
        dm_p2_e = numpy.dot(mo_coeff * pdiag, mo_coeff.T)

        # Frozen orbital response
        if with_frozen:
            dco = Imat[OF[:, None], OA] / (mo_energy[OF, None] - mo_energy[OA])
            dfv = Imat[VF[:, None], VA] / (mo_energy[VF, None] - mo_energy[VA])
            dm1mo[OF[:, None], OA] = dco
            dm1mo[OA[:, None], OF] = dco.T
            dm1mo[VF[:, None], VA] = dfv
            dm1mo[VA[:, None], VF] = dfv.T

        dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
        vhf = _get_self_veff(mp_e._scf, mol_e, dm1) * 2
        Xvo = reduce(numpy.dot, (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]))
        Xvo += Imat[:nocc, nocc:].T - Imat[nocc:, :nocc]

        # ===== 3) Build protonic driving force (electrostatic drag) =====
        Lp = {}
        if dm_p2_e is not None:
            dm_e_dict = {'e': dm_p2_e}
            for (t1, t2), interaction in mf.interactions.items():
                if 'e' not in (t1, t2):
                    continue
                t_p = t2 if t1 == 'e' else t1
                if not t_p.startswith('n'):
                    continue
                vj = interaction.get_vint(dm_e_dict)
                vj_p = vj.get(t_p)
                if vj_p is None:
                    continue
                comp_p = mf.components[t_p]
                mo_occ_p = comp_p.mo_occ
                nocc_p = numpy.count_nonzero(mo_occ_p > 0)
                C_p = comp_p.mo_coeff
                vj_mo = reduce(numpy.dot, (C_p.T, vj_p, C_p))
                Lp[t_p] = vj_mo[nocc_p:, :nocc_p]

        # ===== 4) Solve CNEO Z-vectors =====
        mo1, f1 = _solve_cp_cneo(mf, Xvo, Lp)
        mo1_e = mo1.get('e')
        dm1mo_resp = _mo1_to_dm1(mo1_e, mp_e.mo_occ) 
        dm1mo += dm1mo_resp

        dm_z_e = reduce(numpy.dot, (mo_coeff, dm1mo_resp, mo_coeff.T))
        dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))

        Imat[nocc:, :nocc] = Imat[:nocc, nocc:].T
        im1 = reduce(numpy.dot, (mo_coeff, Imat, mo_coeff.T))

        # ===== 5) Electron MP2 gradient assembly =====
        mf_grad_e = self.components['e']
        hcore_deriv = mf_grad_e.hcore_generator(mol_e)
        s1 = mf_grad_e.get_ovlp(mol_e)

        zeta = _make_zeta(mo_energy, mo_coeff, mp_e.mo_occ, dm1mo)
        zeta += rhf_grad.make_rdm1e(mo_energy, mo_coeff, mp_e.mo_occ)

        dm1_total = dm1 + hf_dm1_e
        p1 = numpy.dot(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)
        vhf_s1occ = reduce(numpy.dot, (p1, _get_self_veff(mp_e._scf, mol_e, dm1+dm1.T), p1))

        dm1p = hf_dm1_e + dm1*2

        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = offsetdic[ia]
            de_e[k] += numpy.einsum('xij,ij->x', s1[:, p0:p1], im1[p0:p1])
            de_e[k] += numpy.einsum('xji,ij->x', s1[:, p0:p1], im1[:, p0:p1])
            h1ao = hcore_deriv(ia)
            de_e[k] += numpy.einsum('xij,ji->x', h1ao, dm1_total)
            de_e[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], zeta[p0:p1])
            de_e[k] -= numpy.einsum('xji,ij->x', s1[:, p0:p1], zeta[:, p0:p1])
            de_e[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], vhf_s1occ[p0:p1]) * 2
            de_e[k] -= numpy.einsum('xij,ij->x', vhf1[k], dm1p)

        # ===== 6) Assemble relaxed densities for inter-component terms =====
        dm_hf = mf.make_rdm1()
        dm_rel = {}
        dm_z = {}
        dm_rel['e'] = hf_dm1_e + dm1
        dm_z['e'] = dm_z_e

        de_p = numpy.zeros_like(de_e)
        for t, comp in mf.components.items():
            if not t.startswith('n'):
                continue
            dm_hf_p = dm_hf[t]
            mo1_p = mo1.get(t)
            dm1mo_p = _mo1_to_dm1(mo1_p, comp.mo_occ)
            dm_z_p = reduce(numpy.dot, (comp.mo_coeff, dm1mo_p, comp.mo_coeff.T))
            dm_rel_p = dm_hf_p + 0.5 * dm_z_p
            dm_rel[t] = dm_rel_p
            dm_z[t] = 0.5 * dm_z_p

            # One-particle part for quantum nuclei
            comp_grad = self.components[t]
            hcore_deriv_p = comp_grad.hcore_generator(comp.mol)

            zeta_p = _make_zeta(comp.mo_energy, comp.mo_coeff, comp.mo_occ, dm1mo_p)
            zeta_p += rhf_grad.make_rdm1e(comp.mo_energy, comp.mo_coeff, comp.mo_occ)
            aoslices_p = comp.mol.aoslice_by_atom()
            for k, ia in enumerate(atmlst):
                p0, p1 = aoslices_p[ia][2:]
                h1ao = hcore_deriv_p(ia)
                de_p[k] += numpy.einsum('xij,ij->x', h1ao, dm_rel_p)

        # ===== 7) Inter-component Coulomb gradient =====
        de_int = numpy.zeros_like(de_e)
        for (t1, t2), interaction in mf.interactions.items():
            comp1 = mf.components[t1]
            comp2 = mf.components[t2]
            mol1 = comp1.mol
            mol2 = comp2.mol
            charge1 = comp1.charge
            charge2 = comp2.charge
            dm_hf1 = dm_hf[t1]
            dm_hf2 = dm_hf[t2]
            dm_rel1 = dm_rel[t1] if t1 in dm_rel else dm_hf1
            dm_z2 = dm_z.get(t2)

            # Ensure electron correlation is always on dm_rel1
            if t2 == 'e' and t1 != 'e':
                t1, t2 = t2, t1
                mol1, mol2 = mol2, mol1
                charge1, charge2 = charge2, charge1
                dm_hf1, dm_hf2 = dm_hf2, dm_hf1
                dm_rel1 = dm_rel['e']
                dm_z2 = dm_z.get(t2)

            de_int += neo_grad.grad_pair_int(mol1, mol2, dm_rel1, dm_hf2,
                                             charge1, charge2, atmlst)
            de_int += neo_grad.grad_pair_int(mol1, mol2, dm_hf1, dm_z2,
                                             charge1, charge2, atmlst)

        de = de_e + de_p + de_int
        self._de_e = de_e
        self._de_p = de_p
        self._de_int = de_int

        de += self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            de = self.symmetrize(de, atmlst)

        self.de = de
        self._finalize()
        log.timer('%s gradients' % self.base.__class__.__name__, *cput0)
        return self.de
